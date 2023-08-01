#!/bin/bash

GPU_ID=0  # TODO

# data
DATASET='s3dis'
SPLIT=0     # TODO
DATA_PATH='../Datasets/S3DIS/blocks_bs1_s1'
SAVE_PATH='./log_s3dis/'

# backbone parameter
NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20
BASE_WIDTHS='[128, 64]'

# model setting
PRETRAIN_CHECKPOINT='./log_scannet/log_pretrain_s3dis_S0'  # TODO
N_WAY=2  # TODO
K_SHOT=1  # TODO
N_QUESIES=1
N_TEST_EPISODES=100

# training parameter
NUM_ITERS=40000
EVAL_INTERVAL=2000
LR=0.001
DECAY_STEP=5000
DECAY_RATIO=0.5

# model parameter
N_SUBPROTOTYPES=100
K_CONNECT=200
SIGMA=1

args=(--phase 'train' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pretrain_checkpoint_path "$PRETRAIN_CHECKPOINT" --use_attention
      --n_subprototypes $N_SUBPROTOTYPES  --k_connect $K_CONNECT
      --sigma $SIGMA  --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K 
      --dgcnn_mlp_widths "$MLP_WIDTHS" --base_widths "$BASE_WIDTHS" 
      --n_iters $NUM_ITERS --eval_interval $EVAL_INTERVAL --batch_size 1
      --lr $LR  --step_size $DECAY_STEP --gamma $DECAY_RATIO --run $RUN
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUESIES --n_episode_test $N_TEST_EPISODES)

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
