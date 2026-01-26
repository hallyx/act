# ACT推理服务器 - ZMQ通信使用说明

## 概述
这是一个基于ZMQ的ACT（Action Chunking Transformer）模型推理服务器，用于实时机器人控制推理。

## 修复内容

### 1. 核心问题修复
- **argparse冲突**: 修改了 `detr/main.py` 中的 `parse_args()` 为 `parse_args([])`，避免解析命令行参数
- **参数要求**: 将 `detr/main.py` 中的required参数改为False
- **导入清理**: 移除了 `inference_server.py` 中未使用的argparse导入

### 2. 通信改进
- **错误处理**: 添加了详细的异常处理（KeyError, PickleError等）
- **响应格式**: 统一使用字典格式返回，包含status和数据
- **调试信息**: 添加了traceback打印便于调试

## 服务器启动

### 后台运行
```bash
cd /home/gpuserver/hx/github/act
nohup /home/gpuserver/miniconda3/envs/aloha/bin/python inference_server.py > server.log 2>&1 &
```

### 前台运行（用于调试）
```bash
cd /home/gpuserver/hx/github/act
/home/gpuserver/miniconda3/envs/aloha/bin/python inference_server.py
```

### 查看日志
```bash
tail -f /home/gpuserver/hx/github/act/server.log
```

### 停止服务器
```bash
pkill -f inference_server.py
```

## 客户端使用示例

### Python客户端代码
```python
import zmq
import pickle
import numpy as np

# 连接到服务器
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# 1. 重置模型状态（开始新的episode时）
socket.send(pickle.dumps('RESET'))
response = pickle.loads(socket.recv())
print(response)  # {'status': 'OK'}

# 2. 发送推理请求
# qpos: base_pos(3) + base_vel(6) + joint_pos(7) + gripper(1) = 17维
base_pos = np.array([...], dtype=np.float32)      # shape: (3,)
base_vel = np.array([...], dtype=np.float32)      # shape: (6,)
joint_pos = np.array([...], dtype=np.float32)     # shape: (7,)
gripper = np.array([...], dtype=np.float32)       # shape: (1,)
qpos = np.concatenate([base_pos, base_vel, joint_pos, gripper])  # shape: (17,)

images = {
    'rgb_main': np.array(...),   # shape: (480, 640, 3), dtype: uint8
    'rgb_left': np.array(...),   # shape: (480, 640, 3), dtype: uint8
    'rgb_under': np.array(...)   # shape: (480, 640, 3), dtype: uint8
}

data = {'qpos': qpos, 'images': images}
socket.send(pickle.dumps(data))
response = pickle.loads(socket.recv())

if response['status'] == 'OK':
    action = response['action']  # shape: (8,)
    print(f"Action: {action}")
else:
    print(f"Error: {response['message']}")

# 清理
socket.close()
context.term()
```

## 数据格式要求

### 输入
- **qpos**: numpy数组, shape=(17,), dtype=float32/float64
  - base_pos: 基座位置 (3维)
  - base_vel: 基座速度 (6维: linear + angular)
  - joint_pos: 左臂关节位置 (7维)
  - gripper: 左臂夹爪状态 (1维)
- **images**: 字典，包含三个相机的图像
  - `'rgb_main'`: numpy数组, shape=(480, 640, 3), dtype=uint8
  - `'rgb_left'`: numpy数组, shape=(480, 640, 3), dtype=uint8
  - `'rgb_under'`: numpy数组, shape=(480, 640, 3), dtype=uint8

### 输出
成功时:
```python
{
    'status': 'OK',
    'action': numpy.ndarray  # shape: (16,) [joint_pos(7) + gripper(1) + joint_vel(7) + gripper_vel(1)]
}
```

错误时:
```python
{
    'status': 'ERROR',
    'message': str  # 错误信息
}
```

## 配置参数

在 `inference_server.py` 中可修改以下参数：

```python
PORT = "5555"                      # ZMQ端口
CKPT_DIR = './ckpt'               # 模型检查点目录
TASK_NAME = 'astrobench_dual_arm' # 任务名称
```

## 注意事项

1. **相机名称**: 必须与训练时使用的相机名称完全一致
2. **状态维度**: qpos必须是17维（与训练配置一致）
   - base_pos(3) + base_vel(6) + joint_pos(7) + gripper(1) = 17
3. **动作维度**: action输出是16维
   - joint_pos(7) + gripper(1) + joint_vel(7) + gripper_vel(1) = 16
4. **时间聚合**: 服务器内部使用temporal aggregation，需要在每个episode开始时发送RESET
5. **内存管理**: 服务器会为每个episode预分配buffer，默认最大1000步

## 测试

运行测试客户端:
```bash
cd /home/gpuserver/hx/github/act
/home/gpuserver/miniconda3/envs/aloha/bin/python test_zmq_client.py
```

预期输出:
```
Connected to ACT inference server on port 5555

--- Test 1: RESET ---
Response: {'status': 'OK'}

--- Test 2: Inference ---
Sending qpos shape: (17,)
Sending images: ['rgb_main', 'rgb_left', 'rgb_under']
Response status: OK
Action shape: (16,)
Action: [...]

--- Test 3: RESET again ---
Response: {'status': 'OK'}

✓ All tests completed!
```

## 故障排除

### 问题: Address already in use
```bash
# 查找占用端口的进程
lsof -i :5555
# 或
netstat -nlp | grep 5555

# 停止进程
kill -9 <PID>
```

### 问题: 模型加载失败
- 检查 `CKPT_DIR` 路径是否正确
- 确认 `policy_best.ckpt` 和 `dataset_stats.pkl` 存在

### 问题: 相机名称错误
- 检查 `constants.py` 中 `astrobench_dual_arm` 的 `camera_names` 配置
- 确保客户端发送的图像字典键名与之匹配

## 性能优化建议

1. **GPU预热**: 第一次推理可能较慢，可以先发送几次测试请求
2. **批处理**: 如果需要处理多个请求，可以考虑修改服务器支持批处理
3. **网络延迟**: 对于远程通信，考虑使用 `tcp://` 而非 `ipc://`
