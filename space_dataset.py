import os
import glob
from typing import List, Dict, Optional
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:
    cv2 = None



aura_state_mean = np.array([ 3.1795579e-01 ,-7.3184729e-01 ,-1.2341137e-01 , 1.8923626e+00,
  3.0520085e-02 ,-1.0634720e+00, -9.9408239e-02, -7.5122500e-03,
 -9.8064477e-03, -9.4816014e-03, -6.0841337e-02 ,-1.9399364e-02,
  1.4716079e-02 , 9.9675453e-01, -1.0356457e-03, -5.6781824e-04,
 -6.6601805e-04, -8.9616254e-03, -3.1446267e-03 , 2.7019314e-03])
aura_state_std = np.array([0.21110618, 0.3252946,  0.08807489, 0.2254707,  0.06606124, 0.08890882,
 0.15063687, 0.01245828, 0.0096468,  0.00710343 ,0.04122881 ,0.01415034,
 0.01658973 ,0.00327328, 0.00363987 ,0.00224044 ,0.00202342, 0.02987257,
 0.00953984, 0.00882406])
aura_action_mean = np.array([ 0.318873 ,  -0.73280454, -0.12367655,  1.8911687 ,  0.03062093 ,-1.0632511,
 -0.09966011,  0.04585912, -0.04786038, -0.01325861, -0.05969343,  0.00504229,
  0.01104373, -0.01259394,0.00])
aura_action_std = np.array([2.1081884e-01, 3.2549274e-01, 8.7964803e-02 ,2.2569728e-01, 6.6079326e-02,
 8.9186713e-02 ,1.5075536e-01, 6.5260842e-02, 7.3913395e-02, 2.8666798e-02,
 7.6001346e-02 ,6.2250912e-02, 5.6798641e-02 ,4.6677053e-02,1])


dawn_state_mean = np.array([ 3.30968827e-01, -7.19925821e-01, -1.31538481e-01,  1.89174449e+00,
  2.74691600e-02, -1.06483483e+00, -1.06749855e-01, -6.88721333e-03,
 -9.29463655e-03, -9.96917766e-03, -6.31241649e-02, -2.02092119e-02,
  1.37933977e-02,  9.96514857e-01, -9.22617561e-04, -4.80249408e-04,
 -7.51718530e-04, -9.90892109e-03, -3.39849060e-03,  2.38162512e-03])
dawn_state_std = np.array([0.22259864, 0.3177003,  0.09153394, 0.22768958, 0.06847755, 0.08879329,
 0.15181772, 0.01184912, 0.0094275 , 0.00753929,0.04340611 ,0.01493294,
 0.01598874, 0.00350697, 0.00344621, 0.00215869, 0.00215438, 0.03155632,
 0.01003225, 0.00831934])
dawn_action_mean = np.array([ 0.33192417, -0.72085243, -0.13182293 , 1.8905473,   0.02756402 ,-1.0646152,
 -0.10702132,  0.0477661,  -0.04633095, -0.01422186, -0.05985721 , 0.00474276,
  0.01097941, -0.01357364,  0.0 ])
dawn_action_std = np.array([2.2230995e-01, 3.1790704e-01 ,9.1408961e-02, 2.2792022e-01, 6.8501547e-02,
 8.9072272e-02, 1.5192370e-01, 6.8286195e-02, 7.2326146e-02, 2.9934566e-02,
 7.6558150e-02, 6.5338805e-02, 5.6660473e-02, 4.8711125e-02, 1.00])


hubble_state_mean = np.array([ 3.3163792e-01, -7.4146241e-01, -1.2986025e-01,  1.8876431e+0,
  2.7686754e-02 ,-1.0615594e+00, -9.9915855e-02, -7.7705788e-03,
 -9.9199899e-03, -9.9912891e-03, -6.3582249e-02, -2.0375652e-02,
  1.4983953e-02 , 9.9647492e-01, -1.0552879e-03 ,-5.5892370e-04,
 -7.5035740e-04 ,-1.0042021e-02, -3.4606257e-03,  2.7322595e-03])
hubble_state_std = np.array([0.22143474 ,0.3205628 , 0.09252706, 0.22916728 ,0.06832495 ,0.08997348,
 0.15790337 ,0.01212312, 0.00951106 ,0.00748273, 0.04313597, 0.01483327,
 0.01620317 ,0.00348903, 0.00361769, 0.00222458, 0.00216924, 0.03180165,
 0.01013365 ,0.00875635])
hubble_action_mean = np.array([ 0.33259526, -0.7424488,  -0.1301412,   1.8864343,   0.0277821,  -1.0613325,
 -0.10016903 , 0.04786809 ,-0.04932226, -0.01404735, -0.06043881,  0.00476724,
  0.0113488 , -0.01265892 , 0.00])
hubble_action_std = np.array([2.2113834e-01, 3.2072949e-01, 9.2412345e-02, 2.2939095e-01, 6.8348989e-02,
 9.0256609e-02 ,1.5803434e-01, 6.8414703e-02,7.3968627e-02, 2.9885657e-02,
 7.6856740e-02, 6.5116867e-02 ,5.7053514e-02, 4.8769590e-02, 1.00])


AURA_STATS = dict(
    state_mean=aura_state_mean, state_std=aura_state_std,
    action_mean=aura_action_mean, action_std=aura_action_std,
)

DAWN_STATS = dict(
    state_mean=dawn_state_mean, state_std=dawn_state_std,
    action_mean=dawn_action_mean, action_std=dawn_action_std,
)

HUBBLE_STATS = dict(
    state_mean=hubble_state_mean, state_std=hubble_state_std,
    action_mean=hubble_action_mean, action_std=hubble_action_std,
)

STATS_MAP = {
    "AURA_STATS": AURA_STATS,
    "DAWN_STATS": DAWN_STATS,
    "HUBBLE_STATS": HUBBLE_STATS,
}



def _ensure_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        return x.astype(np.float32, copy=False)
    return x


def _sanitize_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    std = np.asarray(std, dtype=np.float32)
    std = np.maximum(std, eps)   # 防止除零
    return std


def normalize_state(
    obs_state_list: List[np.ndarray],
    stats: Dict[str, np.ndarray],
) -> List[np.ndarray]:
    """
    obs_state_list: list of (num_obs, 20) float32
    returns: normalized list (same shapes)
    """
    mean = np.asarray(stats["state_mean"], dtype=np.float32).reshape(1, -1)  # (1,20)
    std  = _sanitize_std(stats["state_std"]).reshape(1, -1)                  # (1,20)

    out = []
    for s in obs_state_list:
        s = _ensure_float32(s)
        # (num_obs,20) - (1,20) / (1,20)
        out.append((s - mean) / std)
    return out


def normalize_action_re(
    action_list: List[np.ndarray],
    stats: Dict[str, np.ndarray],
    is_pad_list: Optional[List[np.ndarray]] = None,
    normalize_padded: bool = True,
) -> List[np.ndarray]:
    """
    action_list: list of (num_action, 15) float32
    is_pad_list: list of (num_action,) bool; if provided and normalize_padded=False,
                 pad positions will be left unchanged (raw) OR you can set them to 0.
    normalize_padded:
        - True: pad 位置也做 normalize（简单、最常见）
        - False: pad 位置不做 normalize（更“严格”，需要你下游loss严格mask pad）
    """
    mean = np.asarray(stats["action_mean"], dtype=np.float32).reshape(1, -1)  # (1,15)
    std  = _sanitize_std(stats["action_std"]).reshape(1, -1)                  # (1,15)

    out = []
    for i, a in enumerate(action_list):
        a = _ensure_float32(a)
        if (is_pad_list is None) or normalize_padded:
            out.append((a - mean) / std)
        else:
            # 仅对非 pad normalize
            pad = is_pad_list[i].astype(bool)
            a_norm = a.copy()
            valid = ~pad
            a_norm[valid] = (a_norm[valid] - mean) / std

            # pad 部分怎么处理：两种都合理
            # 方案1：保持原值（a_norm[pad] 不动）
            # 方案2：直接置 0（更常见，且和 mask-loss 配合很好）
            # a_norm[pad] = 0.0

            out.append(a_norm)
    return out



def find_all_hdf5(dataset_dir: str, keyword: Optional[str] = None) -> List[str]:
    patterns = [
        os.path.join(dataset_dir, "**", "*.h5"),
        os.path.join(dataset_dir, "**", "*.hdf5"),
    ]
    all_paths = []
    for p in patterns:
        all_paths.extend(glob.glob(p, recursive=True))
    all_paths = sorted(list(set(all_paths)))
    if keyword is not None:
        all_paths = [p for p in all_paths if keyword in os.path.basename(p)]
    if len(all_paths) == 0:
        raise FileNotFoundError(f"No h5 files found in {dataset_dir} (keyword={keyword})")
    print(f"Found {len(all_paths)} h5 files under {dataset_dir}")
    return all_paths


def decode_jpeg(blob) -> np.ndarray:
    """Decode one JPEG blob (bytes/np.void) -> RGB uint8 (H,W,3)."""
    if cv2 is None:
        raise ImportError("opencv-python is required. pip install opencv-python")

    if isinstance(blob, (bytes, bytearray)):
        buf = np.frombuffer(blob, dtype=np.uint8)
    else:
        # h5py may return np.void/object-like
        buf = np.frombuffer(blob.tobytes(), dtype=np.uint8)

    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode failed (bad bytes?)")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


class H5Dataset(Dataset):
    """
    参考 RealWorldDataset_re 的思路：
    - __init__ 直接把所有 h5 轨迹读出来并生成样本列表
    - __getitem__ 直接从 list 拿样本，不再打开 h5

    Sample:
      obs_rgb:  (num_obs, 2, H, W, 3)  # 2 cameras: main+left
      obs_state:(num_obs, state_dim)   # joint_pos+base_pose+base_vel
      action:   (num_action, action_dim) # jp+jv+gripper
      is_pad:   (num_action,) bool      # padding mask for action horizon
    """

    def __init__(
        self,
        dataset_dir: str,
        num_obs: int = 2,
        num_action: int = 20,
        image_size: int = None,
        normalize_name: bool = False,
        use_next_action: bool = True,  # True: action from t+1..t+num_action
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.num_obs = num_obs
        self.num_action = num_action
        self.image_size = image_size
        self.normalize_name = normalize_name
        self.use_next_action = use_next_action

        # buffers (like RealWorldDataset_re)
        self.obs_rgb_main = []   # each: (num_obs, H, W, 3) uint8
        self.obs_rgb_left = []   # each: (num_obs, H, W, 3) uint8
        self.obs_state = []      # each: (num_obs, state_dim) float32
        self.action = []         # each: (num_action, action_dim) float32
        self.is_pad = []         # each: (num_action,) bool

        h5_paths = find_all_hdf5(dataset_dir, keyword=None)
        q_left_index = [0, 2, 4, 6, 8, 10, 12]  # 左臂关节索引，后续7个是夹爪

        for ep_i, h5_path in enumerate(h5_paths):
            with h5py.File(h5_path, "r") as f:
                # ---- load trajectory length T ----
                if "metadata" in f and "num_frames" in f["metadata"].attrs:
                    T = int(f["metadata"].attrs["num_frames"])
                else:
                    T = int(f["observations"]["rgb_main"].shape[0])

                # ---- read full arrays (trajectory-level) ----
                # observations
                rgb_main_ds = f["observations"]["rgb_main"]   # (T,) jpeg bytes
                rgb_left_ds = f["observations"]["rgb_left"]   # (T,) jpeg bytes

                joint_left = f["observations"]["joint_pos"][:,q_left_index].astype(np.float32)   # (T,7)  左臂角度空间
                #pos_left = f["observations"]["ee_pose"][:, :7].astype(np.float32)   #[:7]    # (T,7)  左臂末端 笛卡尔坐标系
                base_pose = f["observations"]["base_pose"][...].astype(np.float32)   # (T,7)  # 基座位姿 笛卡尔
                base_vel = f["observations"]["base_vel"][...].astype(np.float32)     # (T,6)  # 基座速度

                # actions
                action_left = f["action"]["joint_positions"][:,q_left_index].astype(np.float32)         # (T,7)  左臂的角度空间
                action_vec_left = f["action"]["joint_velocities"][:, q_left_index].astype(np.float32)   # (T,7)  左臂速度
                #action_pos_all = f["action"]["cartesian_target_pose"][:, :8].astype(np.float32)        # 左臂末端 + 夹爪开合  笛卡尔坐标系
                eef_01 = f["action"]["gripper_command"][:, 0].astype(np.float32)                  #  夹爪开合
                # ---- build per-timestep sample like your code ----
                pose_ids = list(range(T))  # [0..T-1]

                # 你原代码是 for cur_idx in range(len(pose_ids)-1)
                # 这里同样遍历每个时刻作为“当前帧”
                for cur_idx in range(len(pose_ids)-2):
                    # obs ids: length=num_obs, pad before with first frame id
                    obs_pad_before = max(0, self.num_obs - cur_idx - 1)
                    frame_begin = max(0, cur_idx - self.num_obs + 1)
                    obs_ids = pose_ids[:1] * obs_pad_before + pose_ids[frame_begin: cur_idx + 1]
                    # obs_ids len == num_obs

                    # action ids: length=num_action, take future steps
                    # 默认 use_next_action=True：预测下一步开始的未来20步（更常见）
                    act_start = cur_idx + 1 if self.use_next_action else cur_idx
                    frame_end = min(len(pose_ids), act_start + self.num_action)
                    action_ids = pose_ids[act_start: frame_end]

                    action_pad_after = self.num_action - len(action_ids)
                    if action_pad_after > 0:
                        action_ids = action_ids + pose_ids[-1:] * action_pad_after  # pad with last

                    # mask for pad
                    is_pad = np.zeros((self.num_action,), dtype=np.bool_)
                    if action_pad_after > 0:
                        is_pad[-action_pad_after:] = True

                    # ---- decode rgb for obs window ----
                    rgb_main = []
                    rgb_left = []
                    for ti in obs_ids:
                        im_main = decode_jpeg(rgb_main_ds[ti])
                        im_left = decode_jpeg(rgb_left_ds[ti])
                        if self.image_size is not None:
                            im_main = cv2.resize(im_main, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                            im_left = cv2.resize(im_left, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                        rgb_main.append(im_main)
                        rgb_left.append(im_left)
                    rgb_main = np.stack(rgb_main, axis=0)  # (num_obs,H,W,3)
                    rgb_left = np.stack(rgb_left, axis=0)  # (num_obs,H,W,3)

                    # ---- state for obs window (aligned with obs_ids) ----
                    # state_dim = 7 + 7 + 6 = 20
                    state = np.concatenate(
                        [joint_left[obs_ids], base_pose[obs_ids], base_vel[obs_ids], ],
                        axis=-1
                    ).astype(np.float32)  # (num_obs,20)

                    # ---- action horizon ----
                    # action_dim = 7 + 7 +1 = 15
                    action = np.concatenate(
                        [action_left[action_ids], action_vec_left[action_ids],  eef_01[action_ids][:, None]],
                        axis=-1
                    ).astype(np.float32)  # (num_action,15)


                    # ---- append to buffers ----
                    self.obs_rgb_main.append(rgb_main)
                    self.obs_rgb_left.append(rgb_left)
                    self.obs_state.append(state)
                    self.action.append(action)
                    self.is_pad.append(is_pad)

            if (ep_i + 1) % 10 == 0:
                print(f"Loaded {ep_i+1}/{len(h5_paths)} episodes, samples so far: {len(self.action)}")
        stats = STATS_MAP[self.normalize_name]
        self.obs_state = normalize_state(self.obs_state, stats=stats)
        self.action = normalize_action_re(
            self.action,
            stats=stats,
            is_pad_list=self.is_pad,
            normalize_padded=True,  # 建议先 True
        )
        print(f"[RealWorldH5Dataset_re] total samples: {len(self.action)}")

    def __len__(self):
        return len(self.action)

    def __getitem__(self, index):
        # obs rgb -> (num_obs, 2, 3, H, W) float32 in [0,1]
        rgb_main = self.obs_rgb_main[index].astype(np.float32) / 255.0  # (num_obs,H,W,3)
        rgb_left = self.obs_rgb_left[index].astype(np.float32) / 255.0  # (num_obs,H,W,3)

        # stack cameras
        # (num_obs, 2, H, W, 3)
        obs_rgb = np.stack([rgb_main, rgb_left], axis=1)
        # to torch: (num_obs, 2, 3, H, W)
        obs_rgb = torch.from_numpy(obs_rgb).permute(0, 1, 4, 2, 3).contiguous()

        obs_state = torch.from_numpy(self.obs_state[index]).float()   # (num_obs,20)
        action = torch.from_numpy(self.action[index]).float()         # (num_action,15)
        is_pad = torch.from_numpy(self.is_pad[index]).bool()          # (num_action,)

        return {
            "obs_rgb": obs_rgb,
            "obs_state": obs_state,
            "action": action,
            "is_pad": is_pad,
        }

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # ====== 你只需要改这里 ======
    dataset_dir = "/data/hb/data_ik/aura/"
    keyword = None          # 例如 "recovery"；不筛选就 None
    num_obs = 2
    num_action = 20
    image_size = None        # None 表示不 resize
    batch_size = 2
    num_workers = 0
    inspect_index = 0
    save_debug_images = False
    # ===========================

    ds = H5Dataset(
        dataset_dir=dataset_dir,
        num_obs=num_obs,
        num_action=num_action,
        image_size=image_size,
        keyword=keyword,
        normalize=False,
        norm_stats=None,
        use_next_action=True,
    )

    print("\n==== Dataset Summary ====")
    print("len(ds) =", len(ds))

    idx = min(max(inspect_index, 0), len(ds) - 1)
    sample = ds[idx]
    obs_rgb = sample["obs_rgb"]
    obs_state = sample["obs_state"]
    action = sample["action"]
    is_pad = sample["is_pad"]

    print("\n==== One Sample (index={}) ====".format(idx))
    print("obs_rgb   :", tuple(obs_rgb.shape), obs_rgb.dtype,
          f"min={obs_rgb.min().item():.4f} max={obs_rgb.max().item():.4f}")
    print("obs_state :", tuple(obs_state.shape), obs_state.dtype,
          f"min={obs_state.min().item():.4f} max={obs_state.max().item():.4f}")
    print("action    :", tuple(action.shape), action.dtype,
          f"min={action.min().item():.4f} max={action.max().item():.4f}")
    print("is_pad    :", tuple(is_pad.shape), is_pad.dtype,
          f"num_pad={int(is_pad.sum().item())}")

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    batch = next(iter(dl))

    print("\n==== One Batch ====")
    print("batch['obs_rgb']  :", tuple(batch["obs_rgb"].shape), batch["obs_rgb"].dtype)
    print("batch['obs_state']:", tuple(batch["obs_state"].shape), batch["obs_state"].dtype)
    print("batch['action']   :", tuple(batch["action"].shape), batch["action"].dtype)
    print("batch['is_pad']   :", tuple(batch["is_pad"].shape), batch["is_pad"].dtype)

    if save_debug_images:
        if cv2 is None:
            raise ImportError("opencv-python is required to save debug images")

        os.makedirs("_debug_imgs", exist_ok=True)

        # obs_rgb: (num_obs, 2, 3, H, W) float in [0,1]
        t = num_obs - 1
        img_main = obs_rgb[t, 0].permute(1, 2, 0).cpu().numpy()  # (H,W,3) RGB
        img_left = obs_rgb[t, 1].permute(1, 2, 0).cpu().numpy()

        img_main_bgr = (img_main * 255.0).clip(0, 255).astype("uint8")[:, :, ::-1]
        img_left_bgr = (img_left * 255.0).clip(0, 255).astype("uint8")[:, :, ::-1]

        out_main = os.path.join("_debug_imgs", f"sample{idx}_t{t}_main.png")
        out_left = os.path.join("_debug_imgs", f"sample{idx}_t{t}_left.png")
        cv2.imwrite(out_main, img_main_bgr)
        cv2.imwrite(out_left, img_left_bgr)

        print(f"\nSaved debug images:\n- {out_main}\n- {out_left}")

