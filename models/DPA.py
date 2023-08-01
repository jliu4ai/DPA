import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import fps

from models.dgcnn import DGCNN
from models.dgcnn_new import DGCNN_semseg
from models.attention import SelfAttention, PrototypeRectification, CrossAttention


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i-1]
            self.convs.append(nn.Sequential(
                              nn.Conv1d(in_dim, params[i], 1),
                              nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs-1:
                x = F.relu(x)
        return x


class DynamicPrototypeAdaptation(nn.Module):
    def __init__(self, args):
        super(DynamicPrototypeAdaptation, self).__init__()
        # self.gpu_id = args.gpu_id
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.n_subprototypes = args.n_subprototypes
        self.k_connect = args.k_connect
        self.sigma = args.sigma

        self.n_classes = self.n_way+1

        if args.use_high_dgcnn:
            self.encoder = DGCNN_semseg(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        else:
            self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)

        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)

        self.feat_dim = 320

        self.conv_1 = nn.Sequential(nn.Conv1d(self.feat_dim, self.feat_dim, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(self.feat_dim),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.transformer1 = PrototypeRectification()
        self.transformer2 = CrossAttention()


    def forward(self, support_x, support_y, query_x, query_y):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points)
        support_feat = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, self.feat_dim, self.n_points)
        query_feat = self.getFeatures(query_x)  # (n_queries, feat_dim, num_points)

        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask)   # 【n-way, k-shot, dim】
        support_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)   #  [n-way, k-shot, dim】

        # prototype learning
        fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, support_bg_feat)
        prototypes = [bg_prototype] + fg_prototypes   # [bg, cls_1, cls2] prototype list

        prototypes_class = torch.stack(prototypes, dim=0).unsqueeze(0).repeat(query_feat.shape[0], 1, 1)  # [n-way, n-class, dim]

        support_feat_ = support_feat.mean(1)
        # prototype refinement with transformer
        prototypes_student = self.transformer1(support_feat_, query_feat, prototypes_class)
        prototypes_teacher = self.transformer2(query_feat, prototypes_student)

        # mask prediction
        prototypes_new = torch.chunk(prototypes_teacher, prototypes_teacher.shape[1], dim=1)
        similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1)) for prototype
                      in prototypes_new]
        query_pred = torch.stack(similarity, dim=1)

        # calculate total loss
        query_loss = self.computeCrossEntropyLoss(query_pred, query_y)

        kl_loss = -0.1*torch.sum(F.log_softmax(prototypes_student, dim=-1)*F.softmax(prototypes_teacher, dim=-1), dim=-1).mean()

        return query_pred, query_loss+kl_loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        feat_level1, feat_level2, _ = self.encoder(x)
        feat_level3 = self.base_learner(feat_level2)
        att_feat = self.att_learner(feat_level2)
        out_feat = torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1)
        out_feat = self.conv_1(out_feat)
        return out_feat

    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            prototype: prototype, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2)
        masked_feat = feat * mask
        prototype = torch.sum(masked_feat, dim=3) / (mask.sum(dim=3) + 1e-5)
        return prototype

    def getPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype = bg_feat.sum(dim=(0,1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype

    def calculateSimilarity_trans(self, feat,  prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[..., None], p=2)**2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)

