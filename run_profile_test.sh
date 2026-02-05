#!/bin/bash
# 快速性能分析测试脚本

echo "开始性能分析测试..."
echo "配置: batch_size=4, 1个epoch, 用于快速定位瓶颈"
echo ""

source ~/miniconda3/bin/activate aloha

python imitate_episodes.py \
    --task_name astrobench_dual_arm \
    --ckpt_dir ./ckpt_profile_test \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 20 \
    --hidden_dim 512 \
    --batch_size 4 \
    --dim_feedforward 3200 \
    --num_epochs 1 \
    --lr 1e-5 \
    --seed 0 2>&1 | tee profile_quick.log

echo ""
echo "==================== 性能分析结果 ===================="
echo "查看详细统计:"
grep -A 30 "性能分析统计" profile_quick.log | tail -25

echo ""
echo "查看epoch耗时:"
grep "总耗时" profile_quick.log

echo ""
echo "完整日志保存在: profile_quick.log"
