#!/bin/bash
# 使用更少的数据训练以避免内存问题

# 激活环境
source ~/miniconda3/bin/activate aloha

# 设置参数
TASK_NAME="astrobench_dual_arm"
CKPT_DIR="./ckpt_ik"
POLICY_CLASS="ACT"
KL_WEIGHT=10
CHUNK_SIZE=20
HIDDEN_DIM=512
BATCH_SIZE=4  # 减小batch size
DIM_FEEDFORWARD=3200
NUM_EPOCHS=2000
LR=1e-5
SEED=0

# 启动训练
python3 imitate_episodes.py \
    --task_name $TASK_NAME \
    --ckpt_dir $CKPT_DIR \
    --policy_class $POLICY_CLASS \
    --kl_weight $KL_WEIGHT \
    --chunk_size $CHUNK_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --batch_size $BATCH_SIZE \
    --dim_feedforward $DIM_FEEDFORWARD \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --seed $SEED
