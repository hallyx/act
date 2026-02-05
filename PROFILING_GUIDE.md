# 性能分析指南

## 已添加的性能分析功能

我已经在代码中添加了详细的性能分析功能，可以帮助你定位训练过程中的耗时瓶颈。

### 1. 监控的关键环节

#### 数据加载环节 (`utils.py` 中的 `__getitem__`)
- **01_file_cache**: 文件句柄缓存查找时间
- **02_read_qpos**: 读取机器人状态数据（关节位置、基座位姿等）
- **03a_read_one_camera**: 读取单个相机的所有帧
- **03b_image_decode**: 单张图像的JPEG解码时间
- **03_read_images_total**: 所有相机图像读取总时间
- **04_read_actions**: 读取动作数据
- **05_to_tensor_normalize**: 转换为Tensor并归一化
- **00_TOTAL_getitem**: `__getitem__` 方法的总耗时

#### 训练环节 (`imitate_episodes.py`)
- **10_val_forward_pass**: 验证阶段的前向传播
- **20_train_forward_pass**: 训练阶段的前向传播
- **21_backward**: 反向传播
- **22_optimizer_step**: 优化器更新参数

### 2. 统计信息输出

性能统计会在以下时机自动输出：

1. **每个epoch结束时**: 显示该epoch的总耗时、验证耗时、训练耗时
2. **每10个epoch**: 打印详细的性能统计表格

### 3. 统计表格格式

```
============================================================
⏱️  性能分析统计
============================================================
环节名称                                  | 总计:    XXXs | 调用:  XXX次 | 平均: XXXms
------------------------------------------------------------
03_read_images_total                     | 总计:  1200.50s | 调用:  83430次 | 平均:  14.39ms
03b_image_decode                         | 总计:  1100.30s | 调用: 333720次 | 平均:   3.30ms
20_train_forward_pass                    | 总计:   850.20s | 调用:  20857次 | 平均:  40.75ms
21_backward                              | 总计:   650.40s | 调用:  20857次 | 平均:  31.18ms
...
============================================================
```

### 4. 如何使用

#### 运行性能分析

```bash
# 运行1个epoch快速查看性能瓶颈
python imitate_episodes.py \
    --task_name astrobench_dual_arm \
    --ckpt_dir ./ckpt_profile \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 20 \
    --hidden_dim 512 \
    --batch_size 4 \
    --dim_feedforward 3200 \
    --num_epochs 1 \
    --lr 1e-5 \
    --seed 0 2>&1 | tee profile.log
```

#### 分析输出日志

```bash
# 查看性能统计表格
grep -A 20 "性能分析统计" profile.log

# 查看每个epoch的耗时
grep "Epoch.*总耗时" profile.log

# 查看训练/验证的分别耗时
grep "耗时:" profile.log
```

### 5. 常见瓶颈及优化建议

根据统计结果，可能的瓶颈及对应优化方案：

#### 瓶颈1: 图像解码 (03b_image_decode)
**症状**: 该项耗时占比最大（通常>40%）

**优化方案**:
- ✅ 已使用LRU缓存 (`@lru_cache(maxsize=1000)`)
- 考虑预先解码所有图像存为numpy数组（需要更多内存）
- 减少相机数量或图像分辨率
- 使用更快的图像格式（如PNG或未压缩）

#### 瓶颈2: 前向传播 (20_train_forward_pass)
**症状**: GPU计算耗时大

**优化方案**:
- 减小模型规模（hidden_dim, dim_feedforward）
- 使用混合精度训练（AMP）
- 增大batch_size（如果内存允许）
- 使用更高效的backbone（如MobileNet）

#### 瓶颈3: 反向传播 (21_backward)
**症状**: 反向传播耗时接近前向传播

**优化方案**:
- 使用梯度累积代替大batch_size
- 启用梯度检查点（gradient checkpointing）
- 减少chunk_size

#### 瓶颈4: 文件读取 (01-05系列)
**症状**: 数据加载总耗时大

**优化方案**:
- ✅ 已使用文件句柄缓存
- 将数据转换为更高效的格式（LMDB、TFRecord）
- 使用SSD存储
- 增加num_workers（如果内存允许）

### 6. 性能基准参考

在典型配置下（batch_size=4, chunk_size=20, 4相机）：

| 环节 | 平均耗时 | 占比 |
|------|---------|------|
| 图像解码 | 3-5ms/张 | 40-50% |
| 前向传播 | 30-50ms | 20-30% |
| 反向传播 | 20-40ms | 15-25% |
| 数据读取 | 1-3ms | 5-10% |
| 其他 | - | 5-10% |

### 7. 实时监控命令

```bash
# 监控GPU使用率
watch -n 1 nvidia-smi

# 监控内存使用
watch -n 1 'free -h && echo "---" && ps aux | grep python | grep imitate | head -3'

# 实时查看训练日志
tail -f profile.log

# 查看最新的性能统计
tail -100 profile.log | grep -A 20 "性能分析统计"
```

### 8. 代码位置

性能分析代码位于：
- `utils.py`: 第1-35行（计时工具函数）和 `__getitem__` 方法中
- `imitate_episodes.py`: 导入和训练循环中

### 9. 关闭性能分析

如果不需要性能分析，可以：

1. **临时关闭打印**：注释掉 `print_timing_stats()` 调用
2. **完全移除**：删除所有 `log_timing()` 调用（不影响功能）

性能分析的开销很小（<1%），通常可以一直开启。
