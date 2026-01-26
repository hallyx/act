# Action维度扩展说明

## 更新日期
2026年1月22日

## 更新内容

### Action维度扩展
Action从**8维**扩展到**16维**，新增关节速度信息：

**旧版本 (8维):**
```
joint_pos(7) + gripper(1) = 8
```

**新版本 (16维):**
```
joint_pos(7) + gripper(1) + joint_vel(7) + gripper_vel(1) = 16
```

### 详细组成
- **joint_pos** (7维): 左臂7个关节的目标位置
- **gripper** (1维): 左臂夹爪的目标位置
- **joint_vel** (7维): 左臂7个关节的目标速度
- **gripper_vel** (1维): 左臂夹爪的目标速度（当前设为0）

## 完整维度总结

### 输入 (State/qpos): 17维
```
base_pos(3) + base_vel(6) + joint_pos(7) + gripper(1) = 17
```

### 输出 (Action): 16维
```
joint_pos(7) + gripper(1) + joint_vel(7) + gripper_vel(1) = 16
```

## 修改的文件

### 1. utils.py
- ✅ `EpisodicDataset.__getitem__()`: 添加joint_velocities到action
- ✅ `get_norm_stats()`: 更新统计计算包含速度维度

### 2. detr/models/detr_vae.py
- ✅ `DETRVAE.__init__()`: 添加action_dim参数，独立于state_dim
- ✅ `build()`: 从args读取state_dim和action_dim
- ✅ `build_cnnmlp()`: 支持动态state_dim和action_dim

### 3. imitate_episodes.py
- ✅ `action_dim`: 新增变量，设为16
- ✅ `policy_config`: 添加action_dim配置
- ✅ `config`: 添加action_dim到训练配置

### 4. inference_server.py
- ✅ `policy_config['action_dim']`: 设为16
- ✅ `self.action_dim`: 存储action维度
- ✅ `all_time_actions`: 使用action_dim而非state_dim
- ✅ 文档字符串: 更新返回值说明

### 5. test_zmq_client.py
- ✅ 测试输出: 详细显示action各部分

### 6. test_qpos_dimensions.py
- ✅ 验证逻辑: 检查action_mean为16维

### 7. 文档
- ✅ ZMQ_SERVER_README.md: 更新所有action维度说明
- ✅ 新增 ACTION_UPDATE_NOTES.md: 本文档

## 数据格式

HDF5文件中已包含所需的速度数据：
- `/action/joint_positions`: (T, 28) - 关节位置
- `/action/joint_velocities`: (T, 28) - 关节速度 ✅
- `/action/gripper_command`: (T, 2) - 夹爪命令

数据加载器会提取：
- 左臂关节位置: `joint_positions[:, [0,2,4,6,8,10,12]]` (7维)
- 左臂关节速度: `joint_velocities[:, [0,2,4,6,8,10,12]]` (7维)
- 左臂夹爪: `gripper_command[:, 0:1]` (1维)
- 夹爪速度: 设为0 (1维)

## 模型架构变化

### 输入层
```python
self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
# state_dim = 17
```

### 输出层
```python
self.action_head = nn.Linear(hidden_dim, action_dim)
# action_dim = 16
```

### 关键改进
- **解耦设计**: state_dim和action_dim独立配置
- **灵活性**: 可以有不同的输入和输出维度
- **向后兼容**: 通过hasattr检查，默认值保持兼容

## 验证测试

运行验证脚本：
```bash
python test_qpos_dimensions.py
```

**预期结果:**
```
✅ qpos_mean shape: (17,) (期望: (17,))
✅ action_mean shape: (16,) (期望: (16,))
✅ 所有测试通过！
```

## 训练配置

确保训练脚本使用正确的维度：

```python
# imitate_episodes.py
state_dim = 17   # 输入维度
action_dim = 16  # 输出维度

policy_config = {
    'state_dim': state_dim,
    'action_dim': action_dim,
    # ... 其他配置
}
```

## 推理配置

确保推理服务器配置匹配：

```python
# inference_server.py
policy_config = {
    'state_dim': 17,   # qpos维度
    'action_dim': 16,  # action维度
    # ... 其他配置
}
```

## Action输出解析

从推理服务器接收到的action可以这样解析：

```python
action = response['action']  # shape: (16,)

# 位置命令
joint_positions = action[0:7]    # 7个关节位置
gripper_position = action[7]     # 夹爪位置

# 速度命令
joint_velocities = action[8:15]  # 7个关节速度
gripper_velocity = action[15]    # 夹爪速度
```

## 物理意义

### 位置+速度控制的优势

1. **更精确的控制**: 同时指定位置和速度目标
2. **平滑轨迹**: 速度信息帮助生成更平滑的运动
3. **动态响应**: 更好地处理动态任务
4. **碰撞避免**: 速度约束有助于安全控制

### 控制模式

可能的控制策略：
- **位置控制**: 仅使用joint_positions和gripper_position
- **速度控制**: 仅使用joint_velocities和gripper_velocity
- **混合控制**: 使用位置作为目标，速度作为前馈

## 性能影响

- **输出维度增加**: 8 → 16 (翻倍)
- **模型参数**: action_head层参数增加
  - 旧: `hidden_dim × 8`
  - 新: `hidden_dim × 16`
  - 增量: 约4KB (假设hidden_dim=512)
- **推理速度**: 基本不变（输出层很小）
- **内存使用**: 略微增加（temporal aggregation buffer增大）

## 兼容性

### ⚠️ 需要重新训练

与之前的state_dim变化类似，action_dim变化也需要：

1. **重新训练模型** (action_head层维度改变)
2. **重新生成统计数据** (dataset_stats.pkl包含16维action)
3. **更新推理服务器配置**

### 不兼容的检查点

旧模型 (action_dim=8) 无法加载到新模型 (action_dim=16)

## 故障排查

### 问题1: 维度不匹配
```
RuntimeError: size mismatch for action_head.weight: expected [8, 512], got [16, 512]
```
**解决**: 使用新训练的checkpoint

### 问题2: 数据加载错误
```
KeyError: '/action/joint_velocities'
```
**解决**: 检查HDF5文件是否包含joint_velocities数据

### 问题3: 统计数据维度错误
```
ValueError: action shape mismatch
```
**解决**: 删除旧的dataset_stats.pkl，重新训练

## 测试清单

- [x] utils.py数据加载包含joint_velocities
- [x] action统计计算为16维
- [x] DETR模型支持独立action_dim
- [x] imitate_episodes.py配置action_dim=16
- [x] inference_server.py配置action_dim=16
- [x] 测试脚本验证16维输出
- [x] 文档更新
- [x] 维度验证测试通过
- [ ] 重新训练模型（待执行）
- [ ] 端到端推理测试（待执行）

## 后续工作

1. **重新训练模型**: 使用新的state_dim=17和action_dim=16
2. **评估性能**: 对比位置+速度 vs 仅位置的控制效果
3. **速度调优**: 可能需要调整速度的归一化范围
4. **夹爪速度**: 考虑从gripper_command计算实际速度而非设为0

## 相关文件

- [utils.py](utils.py) - 数据加载
- [detr/models/detr_vae.py](detr/models/detr_vae.py) - 模型定义
- [imitate_episodes.py](imitate_episodes.py) - 训练脚本
- [inference_server.py](inference_server.py) - 推理服务器
- [test_qpos_dimensions.py](test_qpos_dimensions.py) - 维度验证
- [ZMQ_SERVER_README.md](ZMQ_SERVER_README.md) - 使用文档
- [QPOS_UPDATE_NOTES.md](QPOS_UPDATE_NOTES.md) - qpos更新说明
