import os
import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from runs.eval import test_few_shot
from dataloaders.loader import MyDataset, MyTestDataset, batch_test_task_collate
from models.DPA_learner import DPALearner
from utils.cuda_util import cast_cuda
from utils.logger import init_logger
import time
import wandb


def train(args):
    logger = init_logger(args.log_dir, args)

    # few-shot setup
    setup = args.phase+'i_%s_S%d_N%d_K%d_exprun_%d'% (args.dataset, args.cvfold, args.n_way, args.k_shot, args.run)
    config = {
        "Dataset": args.dataset,
        "Split": args.cvfold,
        "N-way": args.n_way,
        "K-shot": args.k_shot,
        "Run": args.run
    }

    wandb.init(
        project='Few_shot_Point_Cloud_Seg',  # project name
        name=setup,   # experiment name
        config=config
    )

    # init model and optimizer
    DPA = DPALearner(args)

    #Init datasets, dataloaders, and writer
    PC_AUGMENT_CONFIG = {'scale': args.pc_augm_scale,
                         'rot': args.pc_augm_rot,
                         'mirror_prob': args.pc_augm_mirror_prob,
                         'jitter': args.pc_augm_jitter
                         }

    TRAIN_DATASET = MyDataset(args.data_path, args.dataset, cvfold=args.cvfold, num_episode=args.n_iters,
                              n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                              phase=args.phase, mode='train',
                              num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                              pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

    VALID_DATASET = MyTestDataset(args.data_path, args.dataset, cvfold=args.cvfold,
                                  num_episode_per_comb=args.n_episode_test,
                                  n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                                  num_point=args.pc_npts, pc_attribs=args.pc_attribs)
    VALID_CLASSES = list(VALID_DATASET.classes)

    TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=1, collate_fn=batch_test_task_collate)
    VALID_LOADER = DataLoader(VALID_DATASET, batch_size=1, collate_fn=batch_test_task_collate)


    # train
    best_iou = 0
    for batch_idx, (data, sampled_classes) in enumerate(TRAIN_LOADER):
        if torch.cuda.is_available():
            data = cast_cuda(data)

        loss, accuracy = DPA.train(data)
        wandb.log({"Train loss": loss, "Train acc": accuracy})
        if (batch_idx + 1) % 100 == 0:
            logger.cprint('==[Train] Iter: %d | Loss: %.4f |  Accuracy: %f  ==' % (batch_idx, loss, accuracy))

        if (batch_idx+1) % args.eval_interval == 0:

            valid_loss, mean_IoU = test_few_shot(VALID_LOADER, DPA, logger, VALID_CLASSES)
            logger.cprint('\n=====[VALID] Loss: %.4f | Mean IoU: %f  =====\n' % (valid_loss, mean_IoU))
            if mean_IoU > best_iou:
                best_iou = mean_IoU
                save_dict = {'iteration': batch_idx + 1,
                             'model_state_dict': DPA.model.state_dict(),
                             'optimizer_state_dict': DPA.optimizer.state_dict(),
                             'loss': valid_loss,
                             'IoU': best_iou
                             }
                torch.save(save_dict, os.path.join(args.log_dir, 'checkpoint.tar'))

            wandb.log({'mean-IoU': mean_IoU, 'best-IoU': best_iou})

    wandb.finish()