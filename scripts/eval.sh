#!/bin/bash

GPU_ID=0  # TODO

# data
DATASET='s3dis'   # TODO
SPLIT=0  # TODO
DATA_PATH='../Datasets/S3DIS/blocks_bs1_s1' # TODO

# backbone parameters
NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
EDGECONV_WIDTHS='[[64,64], [64, 64], [64, 64]]'
MLP_WIDTHS='[512, 256]'
K=20
BASE_WIDTHS='[128, 64]'

# MAKE SURE IT CORRESPONDS TO SPLIT, N_WAY, K_SHOT.`
MODEL_CHECKPOINT='./log_s3dis/log_mpti_s3dis_S0_N2_K1_Att1'
N_WAY=2  # TODO
K_SHOT=1  # TODO
N_QUESIES=1
N_TEST_EPISODES=100

# model parameters
N_SUBPROTOTYPES=100
K_CONNECT=200
SIGMA=1

args=(--phase 'eval'  --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$MODEL_CHECKPOINT"
      --model_checkpoint_path "$MODEL_CHECKPOINT"
      --n_subprototypes $N_SUBPROTOTYPES  --k_connect $K_CONNECT
      --sigma $SIGMA --use_attention
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS"
      --edgeconv_widths "$EDGECONV_WIDTHS" --dgcnn_k $K 
      --dgcnn_mlp_widths "$MLP_WIDTHS" --base_widths "$BASE_WIDTHS" 
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUESIES --n_episode_test $N_TEST_EPISODES)

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
