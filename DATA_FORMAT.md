0# AstroBench 数据集格式说明
    
## 📋 概述

本数据集用于训练双臂空间机器人的视觉-语言-动作（VLA）模型，包含专家轨迹和恢复轨迹的成对数据，用于学习鲁棒的抓取策略。

**数据采集环境：** Isaac Sim 4.5 + PhysX 物理引擎  
**机器人配置：** 浮动基座 + 双臂（各7自由度）+ 双指夹爪  
**任务类型：** 空间卫星抓取（Peg-in-Hole 变体）  
**控制频率：** 50Hz（物理仿真）/ 10Hz（数据保存）

---

## 📁 文件结构

```
data/
└── {scene_id}/           # 场景ID（如 "20k2", "33k8"）
    ├── pose_config.json  # 卫星-机械臂相对位姿
    ├── T_ks.npy          # 关键点变换矩阵
    └── {lighting_mode}/  # 光照模式（diffuse, hard_sun, earth_albedo）
        ├── episode_00_0000_expert_{lighting}.h5      # 专家轨迹（初次尝试）
        ├── episode_00_0000_recovery_{lighting}.h5    # 恢复轨迹（卫星预放置）
        ├── episode_00_0001_expert_{lighting}.h5
        ├── episode_00_0001_recovery_{lighting}.h5
        └── ...
```

### 命名规则
- `episode_{seq_id:02d}_{episode_id:04d}_{attempt_type}_{lighting}.h5`
  - `seq_id`: CSV目标序列号（0-99）
  - `episode_id`: 累计episode编号（0000-9999）
  - `attempt_type`: `expert` 或 `recovery`
  - `lighting`: 光照模式名称

---

## 📦 HDF5 数据结构

每个 `.h5` 文件包含一条完整轨迹（约30帧），结构如下：

```
episode_00_0000_expert_diffuse.h5
├── metadata/                    # 元数据组
│   ├── @episode_id              # Episode ID (int)
│   ├── @scene_id                # 场景ID (str)
│   ├── @robot_name              # 机器人名称 (str)
│   ├── @num_frames              # 总帧数 (int)
│   ├── @timestamp               # 采集时间戳 (str)
│   ├── @image_encoding          # "jpg_95"
│   ├── @depth_encoding          # "png_uint16"
│   ├── target_pos               # 目标位置 [x, y, z] (3,)
│   ├── target_quat              # 目标姿态 [x, y, z, w] (4,)
│   └── final_pos_error          # 最终位置误差 (float)
│
├── observations/                # 观测数据组（输入特征）
│   ├── rgb_main                 # 主相机RGB (T,) [JPEG压缩]
│   ├── depth_main               # 主相机深度 (T,) [PNG压缩, uint16, 单位mm]
│   ├── rgb_left                 # 左夹爪相机RGB (T,)
│   ├── depth_left               # 左夹爪相机深度 (T,)
│   ├── rgb_right                # 右夹爪相机RGB (T,)
│   ├── depth_right              # 右夹爪相机深度 (T,)
│   ├── rgb_under                # 底部相机RGB (T,)
│   ├── depth_under              # 底部相机深度 (T,)
│   │
│   ├── joint_pos                # 关节位置 (T, 28) [float32, gzip压缩]
│   ├── base_pose                # 基座位姿 (T, 7) [pos(3) + quat(4)]
│   ├── base_vel                 # 基座速度 (T, 6) [linear(3) + angular(3)]
│   ├── ee_pose                  # 末端执行器位姿 (T, 14) [left(7) + right(7)]
│   ├── relative_target_pose     # 相对目标位姿 (T, 7) [可选]
│   ├── delta_ee_target          # 末端-目标偏差 (T, 6) [可选]
│   └── grasp_success            # 抓取成功标志 (T,) [0或1]
│
└── action/                      # 动作数据组（监督标签）
    ├── joint_positions          # 目标关节位置 (T, 28) [关节空间-机器人特定]
    ├── joint_velocities         # 目标关节速度 (T, 28) [关节空间-机器人特定]
    ├── gripper_command          # 夹爪指令 (T, 2) [left, right]
    └── cartesian_target_pose    # 笛卡尔目标位姿 (T, 16) [通用-便于迁移]
```

> **💡 双格式设计说明**：
> - **关节空间** (joint_positions/velocities): 用于当前机器人的精确控制
> - **笛卡尔空间** (cartesian_target_pose): 用于迁移到其他机械臂（通用表示）
> - 两者同时保存，满足不同应用场景的需求
```

---

## 🔍 关键字段说明

### 1. 图像数据（压缩格式）

#### RGB 图像
- **格式：** JPEG 编码（Quality=95）
- **原始分辨率：** 512×512×3 (uint8)
- **存储格式：** 变长字节流 `vlen_dtype(uint8)`
- **解码示例：**
  ```python
  import h5py
  import cv2
  import numpy as np
  
  with h5py.File('episode.h5', 'r') as f:
      encoded_img = f['observations/rgb_main'][0]  # 第0帧
      img_bgr = cv2.imdecode(np.frombuffer(encoded_img, dtype=np.uint8), cv2.IMREAD_COLOR)
      img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 转回RGB
  ```

#### 深度图
- **格式：** PNG 无损编码（16位）
- **单位：** 毫米（mm）
- **原始范围：** 0-65535 (uint16)
- **解码示例：**
  ```python
  encoded_depth = f['observations/depth_main'][0]
  depth_mm = cv2.imdecode(np.frombuffer(encoded_depth, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
  depth_m = depth_mm.astype(np.float32) / 1000.0  # 转换为米
  ```

### 2. 机器人状态（观测）

#### joint_pos (T, 28)
关节位置向量，这个参数直接从isaacsim中读取，包含：
- **[0:6]** 浮动基座（虚拟关节，通常为0）
- **[6,8,10,12,14,16,18]** 左臂7个关节角度（弧度）
- **[7，9,11,13,15,17,19]** 右臂7个关节角度（弧度）
- **[20:28]** 左右夹爪指关节（各4个）
    ```python
    具体格式为：
    self.isaac_joint_names = ['left_base_2_shoulder_joint','right_base_2_shoulder_joint',
                                    'left_shoulder_joint','right_shoulder_joint',
                                    'left_shoulder_2_back_arm_joint','right_shoulder_2_back_arm_joint',
                                    'left_back_2_fore_arm_joint','right_back_2_fore_arm_joint',
                                    'left_arm_2_wrist_joint','right_arm_2_wrist_joint',
                                    'left_wrist_joint','right_wrist_joint',
                                    'left_wrist_2_end_joint','right_wrist_2_end_joint',
                                    'left_hand_index_0_joint','left_hand_middle_0_joint',
                                    'left_hand_thumb_0_joint','right_hand_index_0_joint',
                                    'right_hand_middle_0_joint','right_hand_thumb_0_joint',
                                    'left_hand_index_1_joint','left_hand_middle_1_joint',
                                    'left_hand_thumb_1_joint','right_hand_index_1_joint',
                                    'right_hand_middle_1_joint','right_hand_thumb_1_joint',
                                    'left_hand_thumb_2_joint','right_hand_thumb_2_joint']
    ```
#### base_pose (T, 7)
- **[0:3]** 基座位置 [x, y, z] (米)
- **[3:7]** 基座姿态四元数 [x, y, z, w]

#### ee_pose (T, 14)
双臂末端执行器的世界坐标系位姿：
- **[0:7]** 左臂末端：pos(3) + quat_xyzw(4)
- **[7:14]** 右臂末端：pos(3) + quat_xyzw(4)
- **注意：** 已包含工具偏移 `tool_offset=[0, 0.04, 0]`

#### grasp_success (T,)
- **值：** 0（抓取失败）或 1（抓取成功）
- **用途：** 轨迹质量过滤，训练时可加权或筛选
- **备注：** 这个检验受限于代码，不完全可靠，不可靠的数据都被删除了，保留的都是1

### 3. 动作数据（监督标签）

**双格式设计**：同时保存关节空间和笛卡尔空间数据

#### A. 关节空间表示（Joint-space）- 机器人特定

##### joint_positions (T, 28)
目标关节位置命令（pos_target from HybridController）：
- **单位：** 弧度
- **来源：** HybridController的输出，实际发送给Isaac Sim的位置指令
- **用途：** 精确复现当前机器人的控制行为
- **关键点：** 这是`current_pos + velocity * lookahead_time`的结果，不是简单的FK积分

##### joint_velocities (T, 28)
目标关节速度命令（vel_target from HybridController）：
- **单位：** 弧度/秒
- **来源：** HybridController的输出，实际发送给Isaac Sim的速度指令
- **用途：** 速度前馈控制，提高轨迹跟踪精度
- **注意：** 由于lookahead机制，`vel_target ≠ (pos[t+1] - pos[t]) / dt` 是正常的

##### gripper_command (T, 2)
夹爪控制指令：
- **格式：** [left_gripper, right_gripper]
- **值：** 0.0（开启）或 1.0（闭合）
- **用途：** 双臂独立的夹爪控制

#### B. 笛卡尔空间表示（Cartesian-space）- 通用表示

##### cartesian_target_pose (T, 16)
下一帧的目标末端位姿（便于跨机器人迁移）：
- **[0:8]** 左臂：pos(3) + quat(4) + gripper(1)
- **[8:16]** 右臂：pos(3) + quat(4) + gripper(1)
- **计算方式：** 基于pos_target的Pinocchio FK（而非积分）
- **用途：** 
  - 迁移到其他机械臂（只需重新IK求解）
  - 任务级别的策略学习（与具体机器人解耦）
  - 可视化和分析（更直观）

#### 两种表示的对比

| 特性 | 关节空间 | 笛卡尔空间 |
|------|---------|-----------|
| **精确性** | ✅ 完全精确（直接控制指令） | ⚠️ 需IK求解（可能多解） |
| **通用性** | ❌ 机器人特定 | ✅ 可迁移到其他机械臂 |
| **维度** | 28维（随机器人自由度变化） | 16维（固定：双臂位姿+夹爪） |
| **应用** | 当前机器人的精确控制 | 跨机器人迁移、任务级策略 |

**建议使用场景**：
- **训练当前机器人**：使用joint_positions/velocities（精确）
- **迁移到其他机械臂**：使用cartesian_target_pose（通用）
- **混合策略**：同时学习两种表示，运行时根据场景选择

---

## 🎯 数据特点

### 成对轨迹设计
每个目标点包含两条轨迹：

| 类型 | 初始状态 | 目的 |
|------|---------|------|
| **Expert** | 卫星在目标位置 | 学习从零开始的完整抓取流程 |
| **Recovery** | 卫星在第三阶段终点 | 学习从接近状态恢复，提高鲁棒性 |

### 三阶段轨迹
专家和恢复轨迹均采用三阶段 Peg-in-Hole 规划：
1. **Approach (3s)** - 快速接近进近点（距目标20cm）
2. **Lateral Align (2s)** - 横向对齐修正漂移（无前进）
3. **Insert (2s)** - 直线插入抓取点

### 光照条件
随机光照，模拟太阳从随机角度照射，光强也是随机

---

## 🔧 数据加载示例

### 完整示例（PyTorch DataLoader）

```python
import h5py
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

class AstroBenchDataset(Dataset):
    def __init__(self, data_dir, camera='main', transform=None):
        """
        Args:
            data_dir: 数据根目录
            camera: 使用的相机 ('main', 'left', 'right', 'under')
            transform: 图像预处理（torchvision.transforms）
        """
        self.data_dir = Path(data_dir)
        self.camera = camera
        self.transform = transform
        
        # 收集所有HDF5文件
        self.files = sorted(self.data_dir.rglob('*.h5'))
        print(f"Found {len(self.files)} episodes")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        h5_path = self.files[idx]
        
        with h5py.File(h5_path, 'r') as f:
            # 1. 解码图像
            num_frames = f['metadata'].attrs['num_frames']
            rgb_key = f'observations/rgb_{self.camera}'
            
            rgb_frames = []
            for t in range(num_frames):
                encoded = f[rgb_key][t]
                img = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb_frames.append(img)
            
            rgb_frames = np.stack(rgb_frames)  # (T, H, W, 3)
            
            # 2. 读取状态
            joint_pos = f['observations/joint_pos'][:]      # (T, 28)
            ee_pose = f['observations/ee_pose'][:]          # (T, 14)
            grasp_success = f['observations/grasp_success'][:] # (T,)
            
            # 3. 读取动作标签
            action_vel = f['action/joint_vel'][:]           # (T, 28)
            action_pose = f['action/cartesian_target_pose'][:] # (T, 16)
        
        # 4. 数据预处理
        if self.transform:
            rgb_frames = self.transform(rgb_frames)
        
        return {
            'rgb': torch.from_numpy(rgb_frames).float(),
            'joint_pos': torch.from_numpy(joint_pos).float(),
            'ee_pose': torch.from_numpy(ee_pose).float(),
            'action_vel': torch.from_numpy(action_vel).float(),
            'action_pose': torch.from_numpy(action_pose).float(),
            'grasp_success': torch.from_numpy(grasp_success).float()
        }

# 使用示例
dataset = AstroBenchDataset('data/20k2/diffuse')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

for batch in dataloader:
    rgb = batch['rgb']              # (B, T, H, W, 3)
    action = batch['action_vel']    # (B, T, 28)
    # ... 训练代码
```

### 快速查看元数据

```python
import h5py

def inspect_episode(h5_path):
    with h5py.File(h5_path, 'r') as f:
        print("=== Metadata ===")
        for key, val in f['metadata'].attrs.items():
            print(f"  {key}: {val}")
        
        print("\n=== Data Shapes ===")
        print(f"  RGB frames: {len(f['observations/rgb_main'])}")
        print(f"  Joint pos: {f['observations/joint_pos'].shape}")
        print(f"  Actions: {f['action/joint_vel'].shape}")
        
        print("\n=== Grasp Success ===")
        grasp = f['observations/grasp_success'][:]
        print(f"  Final result: {'Success' if grasp[-1] == 1 else 'Failure'}")

inspect_episode('data/20k2/diffuse/episode_00_0000_expert_diffuse.h5')
```

---

## 📊 数据统计

### 标准数据集配置
- **场景数量：** 9个不同的卫星姿态
- **每场景目标：** 100个成功的抓取点
- **每目标轨迹：** 2条（expert + recovery）
- **总轨迹数：** ~900条
- **总帧数：** ~90,000帧
- **磁盘空间：** ~10GB（压缩后）

### 单条轨迹统计
- **持续时间：** ~10秒（6+2+2阶段）
- **帧数：** ~104帧（10Hz采样）
- **文件大小：** ~6-9MB（压缩后）
- **原始大小：** ~10-20MB（未压缩）

---


## 🐛 常见问题

### Q1: 图像解码失败？
**A:** 确保使用 `cv2.imdecode` 而非 `cv2.imread`，HDF5中存储的是编码后的字节流。

### Q2: 深度值异常？
**A:** 深度图单位是毫米（mm），需除以1000转换为米。无效深度标记为0。

### Q3: 四元数归一化？
**A:** 数据集中的四元数已归一化，但网络预测时需要手动归一化输出。

### Q4: Recovery轨迹与Expert有何不同？
**A:** Recovery的初始卫星位置在第三阶段终点，机械臂仍从初始位置出发，模拟"接近后恢复"场景。

### Q5: 如何处理多相机数据？
**A:** 可拼接多视角特征或用注意力机制融合。建议主相机（main）+ 左夹爪相机（left）组合。

---

**最后更新：** 2026-01-20  
**数据版本：** v1.0  
**许可协议：** MIT License
