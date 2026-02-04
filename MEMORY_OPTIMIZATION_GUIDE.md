# 内存优化指南

## 问题描述

训练过程中出现"已杀死" (Killed) 错误，通常是系统内存不足导致OOM Killer杀死进程。

## 已实施的优化

### 1. DataLoader优化
- **num_workers=0**: 使用单进程数据加载，避免多进程内存开销
- **pin_memory=False**: 关闭内存固定，减少内存占用
- **验证集比例从80%降至90%**: 减少验证集大小

### 2. 训练循环优化
在 `imitate_episodes.py` 中添加了：
- 每50个batch清理一次GPU缓存（验证阶段）
- 每100个batch清理一次GPU缓存（训练阶段）
- 每个epoch结束后执行垃圾回收 `gc.collect()`
- 及时删除中间变量 `del data, forward_dict, loss`

### 3. Batch Size调整
从24 → 16 → 8 → 4逐步降低

## 推荐配置

### 方案1：最小内存配置（推荐）
```bash
python3 imitate_episodes.py \
    --task_name astrobench_dual_arm \
    --ckpt_dir ./ckpt_ik \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 20 \
    --hidden_dim 512 \
    --batch_size 2 \  # 进一步减小
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0
```

### 方案2：使用更少的数据
修改 `utils.py` 中的 `load_data` 函数，限制使用的文件数量：

```python
# 在 load_data 函数开始处添加
if num_episodes < len(all_file_paths):
    all_file_paths = all_file_paths[:num_episodes]
```

### 方案3：使用梯度累积
当batch_size=2时训练太慢，可以使用梯度累积：

```python
# 在训练循环中
accumulation_steps = 4  # 相当于batch_size=8
for batch_idx, data in enumerate(train_dataloader):
    forward_dict = forward_pass(data, policy)
    loss = forward_dict['loss'] / accumulation_steps
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 监控内存使用

### 实时监控
```bash
watch -n 1 'free -h && echo "---" && nvidia-smi'
```

### 检查进程内存
```bash
ps aux | grep python | grep imitate
```

### 查看日志
```bash
tail -f training_*.log
```

## 如果仍然被杀死

### 1. 增加系统交换空间
```bash
# 创建8GB交换文件
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 2. 使用分布式训练
如果有多个GPU，使用DataParallel或DistributedDataParallel

### 3. 转换数据格式
将HDF5数据预处理为更内存友好的格式（如LMDB或TFRecord）

### 4. 使用混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data in dataloader:
    with autocast():
        loss = model(data)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 当前状态

根据最新测试：
- batch_size=8: 4-5个epoch后OOM
- batch_size=4: 需要进一步测试
- 建议：batch_size=2 或使用梯度累积

## 快速启动

使用提供的脚本：
```bash
./train_reduced.sh
```
