# Motus Evaluation Flow 详解

## 目录

1. [总体流程概览](#1-总体流程概览)
2. [启动入口与参数传递](#2-启动入口与参数传递)
3. [Expert Check 阶段](#3-expert-check-阶段)
4. [仿真数据获取 (get_obs)](#4-仿真数据获取-get_obs)
5. [观测数据处理 (update_obs)](#5-观测数据处理-update_obs)
6. [语言指令处理](#6-语言指令处理)
7. [模型架构](#7-模型架构)
8. [推理流程 (inference_step)](#8-推理流程-inference_step)
9. [Action 输出与反归一化](#9-action-输出与反归一化)
10. [Action 执行 (take_action)](#10-action-执行-take_action)
11. [评测循环总结](#11-评测循环总结)
12. [关键配置参数](#12-关键配置参数)

---

## 1. 总体流程概览

```
debug_single.sh
  └─> python script/eval_policy.py
        ├─ parse_args_and_config()  ← 合并 deploy_policy.yml + CLI overrides
        ├─ get_model(usr_args)      ← 初始化 MotusPolicy (加载模型+权重)
        └─ eval_policy() 循环:
             for each episode (seed):
               ├─ Expert Check: TASK_ENV.play_once() 验证 seed 可行
               ├─ reset_model(model)
               └─ while step < step_limit:
                    ├─ observation = TASK_ENV.get_obs()
                    ├─ eval(TASK_ENV, model, observation):
                    │    ├─ model.set_instruction(instruction)
                    │    ├─ model.update_obs(obs)      ← 图像拼接+resize+state提取
                    │    ├─ actions = model.get_action() ← T5编码+VLM编码+扩散推理
                    │    └─ for action in actions:
                    │         TASK_ENV.take_action(action, action_type='qpos')
                    └─ check success / fail
```

---

## 2. 启动入口与参数传递

### Shell 脚本: `debug_single.sh`

- **工作目录**: `/gemini/code/robotwin`
- **Conda 环境**: `/gemini/code/envs/robotwin`
- **关键参数**:
  - `CHECKPOINT_PATH`: `/gemini/code/models/motus_pretrained_models/Motus_robotwin2`
  - `WAN_PATH`: `.../Wan2.2-TI2V-5B` (WAN 视频模型预训练权重 + T5 编码器)
  - `VLM_PATH`: `.../Qwen3-VL-2B-Instruct` (VLM processor 的 tokenizer)
  - `TASK_CONFIG`: `demo_randomized` (任务配置 yml)
  - `POLICY_NAME`: `Motus`

### 参数解析: `parse_args_and_config()`

1. 加载 `policy/Motus/deploy_policy.yml` 作为 base config
2. CLI `--overrides` 后的 `--key value` 对覆盖 config
3. 最终 `usr_args` dict 包含: `task_name`, `task_config`, `ckpt_setting`, `seed`, `policy_name`, `wan_path`, `vlm_path`, `log_dir`, `instruction_type=unseen`

### 模型初始化: `get_model(usr_args)`

实例化 `MotusPolicy`，内部依次:
1. 加载 `utils/robotwin.yml` 模型配置
2. 创建 `Motus` 模型 (不加载 WAN/VLM 预训练权重，`load_pretrained_backbones=False`)
3. 从 checkpoint 加载完整权重: `model.load_checkpoint(checkpoint_path)`
4. 初始化 T5 编码器 (`umt5-xxl`, bf16)
5. 初始化 VLM processor (Qwen3-VL-2B tokenizer)
6. 加载归一化统计量 (`utils/stat.json`)

---

## 3. Expert Check 阶段

**目的**: 验证当前随机 seed 下，scripted expert 能否完成任务。排除物理不稳定的布局。

```python
# eval_policy.py L220-254
TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
episode_info = TASK_ENV.play_once()  # 硬编码的专家策略
TASK_ENV.close_env()

if TASK_ENV.plan_success and TASK_ENV.check_success():
    # seed 可用，进入 policy 评测
else:
    now_seed += 1  # 跳过该 seed
```

- `play_once()` 是每个 task 类中**硬编码的运动原语序列**
- 只有 expert 能成功的 seed 才会用于 policy 评测
- 测试总数: 100 个有效 episode

---

## 4. 仿真数据获取 (get_obs)

**文件**: `envs/_base_task.py` → `get_obs()` (L437-500)

### 返回的 observation dict 结构

```python
{
    "observation": {
        "head_camera": {
            "rgb": np.ndarray,     # [H, W, 3], uint8, D435相机 (640x480)
            # 以及 intrinsic, extrinsic 等 config
        },
        "left_camera": {
            "rgb": np.ndarray,     # 左腕相机 RGB
        },
        "right_camera": {
            "rgb": np.ndarray,     # 右腕相机 RGB
        },
    },
    "joint_action": {
        "left_arm": list,          # 左臂关节角 (6 dim)
        "left_gripper": float,     # 左夹爪开合 (0=闭, 1=开)
        "right_arm": list,         # 右臂关节角 (6 dim)
        "right_gripper": float,    # 右夹爪开合
        "vector": np.ndarray,      # 拼接: [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)] = 14 dim
    },
    "endpose": {
        "left_endpose": ...,       # 左臂末端位姿
        "right_endpose": ...,      # 右臂末端位姿
        ...
    },
    "pointcloud": [],
}
```

### 关键数据类型 (由 `demo_randomized.yml` 控制)

| 数据类型 | 启用 | 说明 |
|---------|------|------|
| `rgb` | ✅ | 三个相机的 RGB 图像 |
| `qpos` | ✅ | 关节角度 (14 dim vector) |
| `endpose` | ✅ | 末端位姿 |
| `depth` | ❌ | 深度图 |
| `pointcloud` | ❌ | 点云 |

### 相机配置

- **Head Camera**: D435, 固定后方俯视, 640×480
- **Wrist Cameras**: D435, 左右腕各一个, 640×480

---

## 5. 观测数据处理 (update_obs)

**文件**: `deploy_policy.py` → `MotusPolicy.update_obs()` (L173-222)

### 5.1 图像处理流程

```
原始输入:
  head_camera:  [480, 640, 3]  (D435)
  left_camera:  [480, 640, 3]  (D435)
  right_camera: [480, 640, 3]  (D435)

Step 1: 腕部相机 resize
  left_camera  → cv2.resize → [120, 160, 3]
  right_camera → cv2.resize → [120, 160, 3]

Step 2: 拼接为合成图像
  bottom_row = concat([left_resized, right_resized], axis=1) → [120, 320, 3]
  image = concat([head_camera, bottom_row], axis=0)          → [600, 640, 3]
  
  视觉布局:
  ┌──────────────────────┐
  │    Head Camera        │  480 × 640
  │    (固定后视角)        │
  ├──────────┬───────────┤
  │  Left    │  Right    │  120 × 320
  │  Wrist   │  Wrist    │
  └──────────┴───────────┘

Step 3: 转为 tensor
  [600, 640, 3] → permute → [3, 600, 640] → unsqueeze → [1, 3, 600, 640]

Step 4: resize_with_padding 到目标尺寸
  目标: (384, 320)  (来自 robotwin.yml: video_height=384, video_width=320)
  - 计算缩放比: min(384/600, 320/640) = min(0.64, 0.5) = 0.5
  - 缩放后: [300, 320]
  - 黑边填充: 上下各 42px → [384, 320]
  - 归一化: uint8 / 255.0 → float32 [0, 1]

最终: image_tensor [1, 3, 384, 320] on GPU
```

### 5.2 状态处理

```python
state = observation['joint_action']['vector']  # np.ndarray, shape (14,)
# 内容: [left_arm_j1..j6, left_gripper, right_arm_j1..j6, right_gripper]

state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # [1, 14]
self.current_state = state_tensor.to(device)

# 归一化到 [0, 1]
self.current_state_norm = self._normalize_actions(self.current_state)
# normalize: (x - action_min) / (action_max - action_min)
```

### 5.3 归一化统计量 (stat.json)

来自 `utils/stat.json` → `robotwin2` key:
- `min`: 14 维向量, 每个关节/夹爪的最小值
- `max`: 14 维向量, 每个关节/夹爪的最大值
- 统计自 26,649 个训练文件

---

## 6. 语言指令处理

### 6.1 指令来源

```python
# eval_policy.py L260-262
results = generate_episode_descriptions(task_name, episode_info_list, test_num)
instruction = np.random.choice(results[0][instruction_type])  # instruction_type = "unseen"
TASK_ENV.set_instruction(instruction=instruction)
```

- 每个 task 有预定义的 language description 模板
- `instruction_type="unseen"` 使用评测专用的指令变体

### 6.2 T5 编码 (WAN 的文本条件)

```python
# deploy_policy.py L235-245
scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                "The aloha robot is currently performing the following task: ")
instruction = f"{scene_prefix}{self.current_instruction}"

t5_out = self.t5_encoder([instruction], self.device)
# T5 encoder: umt5-xxl, bf16
# 输出: [1, seq_len, 4096] → 后续在模型内通过 text_embedding 层映射到 3072 维
```

### 6.3 VLM 编码 (Understanding Expert 的输入)

```python
# deploy_policy.py L248-311
# 构造 Qwen3-VL 格式的 messages
messages = [{'role': 'user', 'content': [
    {'type': 'text', 'text': instruction},
    {'type': 'image', 'image': first_frame_pil},  # 当前帧 PIL Image
]}]

# Processor tokenize
text = vlm_processor.apply_chat_template(messages, ...)
encoded = vlm_processor(text=[text], images=[image], return_tensors='pt')

vlm_inputs = {
    'input_ids': ...,        # [1, seq_len]
    'attention_mask': ...,   # [1, seq_len]
    'pixel_values': ...,     # [1, num_patches, patch_dim]
    'image_grid_thw': ...,   # [1, 3] - temporal/height/width grid
}
```

---

## 7. 模型架构

### 7.1 总体架构: Motus (三模态 UniDiffuser)

```
Motus
├── Video Module (WAN 2.2 TI2V-5B)
│   ├── VAE Encoder/Decoder (Wan2.2_VAE)
│   ├── Patch Embedding (48ch → 3072D)
│   ├── 30× WanBlock (self-attn + cross-attn + FFN)
│   ├── T5 Text Embedding (4096 → 3072)
│   └── Time Embedding + Projection
│
├── Understanding Module (VLM + Understanding Expert)
│   ├── Qwen3-VL-2B (frozen, 用于提取视觉-语言特征)
│   ├── VLM Adapter (2048 → 512, mlp3x_silu)
│   └── 30× UndExpertBlock (joint-attn projections + FFN)
│
├── Action Module (Action Expert)
│   ├── State Encoder (14 → 1024, mlp3x_silu)
│   ├── Action Encoder (14 → 1024, mlp3x_silu)
│   ├── 4× Register Tokens (可学习, 1024D)
│   ├── Sinusoidal Position Embedding
│   ├── Time Embedding + Projection
│   ├── 30× ActionExpertBlock (joint-attn projections + FFN)
│   └── Action Decoder (1024 → 14, mlp1x_silu + AdaLN)
│
└── Flow-Matching Scheduler (shift=5.0, 1000 train timesteps)
```

### 7.2 MoT (Mixture of Tokens) 三模态联合注意力

**核心设计**: 三种模态的 tokens 在 WAN 的 self-attention 空间中进行联合注意力。

每一层的处理流程:
```
Layer i:
  1. AdaLN Modulation: 计算 video/action 的 6 组调制参数
  
  2. Trimodal Joint Self-Attention (在 WAN 的注意力空间):
     - Video tokens:  norm1 + AdaLN → WAN 原生 Q/K/V (3072D, 24 heads × 128 dim)
     - Action tokens: norm1 + AdaLN → wan_action_qkv 映射 (1024 → 24×128 = 3072D)
     - Und tokens:    norm1         → wan_und_qkv 映射 (512 → 24×128 = 3072D)
     - 拼接所有 Q/K/V → 统一 flash attention
     - 各自投影回原始维度 + 残差连接
  
  3. WAN Cross-Attention (仅 Video):
     - Video tokens × T5 context (3072D) → cross attention → 残差
  
  4. 独立 FFN:
     - Video:  WAN FFN (3072 → 8192 → 3072) + AdaLN
     - Action: Action FFN (1024 → 4096 → 1024) + AdaLN
     - Und:    Und FFN (512 → 2048 → 512) + LayerNorm
```

### 7.3 各组件维度

| 组件 | 隐藏维度 | 注意力头数 | Head Dim | FFN 维度 | 层数 |
|------|---------|-----------|----------|---------|------|
| WAN (Video) | 3072 | 24 | 128 | 8192 | 30 |
| Action Expert | 1024 | (借用WAN的24头) | 128 | 4096 | 30 |
| Understanding Expert | 512 | (借用WAN的24头) | 128 | 2048 | 30 |
| VLM (Qwen3-VL-2B) | 2048 | frozen | - | - | - |
| T5 (umt5-xxl) | 4096 | frozen | - | - | - |

### 7.4 参数量

- **WAN Video Model**: ~5B
- **VLM (Qwen3-VL-2B, frozen)**: ~2B
- **Action Expert**: ~数百M
- **Understanding Expert**: ~数十M

---

## 8. 推理流程 (inference_step)

**文件**: `models/motus.py` → `Motus.inference_step()` (L917-1039)

### 8.1 初始化

```python
# 1. 编码条件帧
first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [0,1] → [-1,1], [1, 3, 1, H, W]
condition_frame_latent = self.video_model.encode_video(first_frame_norm)
# VAE 输出: [1, 48, 1, H', W']  (H'=384/32=12, W'=320/32=10)

# 2. 初始化 video latent (噪声)
num_total_latent_frames = 1 + num_video_frames // 4  # 1 + 8//4 = 3
video_latent = randn(1, 48, 3, 12, 10)
video_latent[:, :, 0:1] = condition_frame_latent  # Teacher Forcing: 第一帧固定

# 3. 初始化 action latent (噪声)
action_chunk_size = num_video_frames * video_action_freq_ratio  # 8 × 2 = 16
action_latent = randn(1, 16, 14)  # [B, chunk_size, action_dim]
```

### 8.2 特征提取 (一次性)

```python
# Understanding Expert: VLM 提取 → adapter 映射
und_tokens = self.und_module.extract_und_features(vlm_inputs)
# VLM forward → last hidden state [1, seq_len, 2048]
# vlm_adapter → [1, seq_len, 512]

# T5 context 预处理
processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)
# T5 output [1, 512, 4096] → text_embedding layer → [1, 512, 3072]
```

### 8.3 去噪循环 (Euler 积分)

```python
timesteps = linspace(1.0, 0.0, num_inference_steps + 1)  # 10 steps: [1.0, 0.9, ..., 0.0]

for i in range(num_inference_steps):  # 10 iterations
    t = timesteps[i]       # 当前时间
    t_next = timesteps[i+1]  # 下一时间
    dt = t_next - t          # 步长 (负数, 约 -0.1)
    
    # 缩放到 WAN 的时间域 [0, 1000]
    video_t_scaled = (t * 1000).expand(B)
    action_t_scaled = (t * 1000).expand(B)
    
    # Tokenize
    video_tokens = video_module.prepare_input(video_latent)
    # patch_embedding(video_latent) → flatten → [1, T'×H'×W'÷4, 3072]
    # 对于 (48, 3, 12, 10): seq_len = 3×12×10÷4 = 90
    
    state_tokens = state.unsqueeze(1)  # [1, 1, 14]
    registers = action_expert.registers.expand(1, -1, -1)  # [1, 4, 1024]
    action_tokens = action_expert.input_encoder(state_tokens, action_latent, registers)
    # StateActionEncoder:
    #   state_encoder(state_tokens) → [1, 1, 1024]
    #   action_encoder(action_latent) → [1, 16, 1024]
    #   concat([state, action, registers]) → [1, 21, 1024]  (1+16+4=21)
    #   + sinusoidal positional embedding
    
    # Re-extract understanding features (每步重新提取)
    und_tokens = und_module.extract_und_features(vlm_inputs)  # [1, seq_len, 512]
    
    # Time embeddings
    video_head_time_emb, video_adaln_params = video_module.get_time_embedding(...)
    action_head_time_emb, action_adaln_params = action_module.get_time_embedding(...)
    
    # 30 层 Transformer
    for layer_idx in range(30):
        # AdaLN modulation
        video_adaln_mod = video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
        action_adaln_mod = action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
        
        # Trimodal Joint Self-Attention
        video_tokens, action_tokens, und_tokens = video_module.process_joint_attention(
            video_tokens, action_tokens, video_adaln_mod, action_adaln_mod, layer_idx,
            action_expert.blocks[layer_idx], und_tokens, und_expert.blocks[layer_idx]
        )
        
        # WAN Cross-Attention with T5
        video_tokens = video_module.process_cross_attention(
            video_tokens, video_adaln_params, layer_idx, processed_t5_context
        )
        
        # Independent FFNs
        video_tokens = video_module.process_ffn(video_tokens, video_adaln_mod, layer_idx)
        action_tokens = action_module.process_ffn(action_tokens, action_adaln_mod, layer_idx)
        und_tokens = und_module.process_ffn(und_tokens, layer_idx)
    
    # Output Heads
    video_velocity = video_module.apply_output_head(video_tokens, video_head_time_emb)
    # WAN head + unpatchify → [1, 48, 3, 12, 10]
    
    action_pred_full = action_expert.decoder(action_tokens, action_head_time_emb)
    # ActionDecoder: AdaLN modulation + action_head MLP → [1, 21, 14]
    action_velocity = action_pred_full[:, 1:-4, :]  # 去掉 state token(1个) 和 registers(4个)
    # → [1, 16, 14]
    
    # Euler Integration
    video_latent = video_latent + video_velocity * dt
    action_latent = action_latent + action_velocity * dt
    
    # Teacher Forcing: 保持条件帧不变
    video_latent[:, :, 0:1] = condition_frame_latent
```

### 8.4 解码输出

```python
# Video 解码
decoded_frames = video_model.decode_video(video_latent)
predicted_frames = decoded_frames[:, :, 1:]  # 跳过条件帧
predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] → [0,1]
predicted_frames = clamp(predicted_frames, 0, 1)
# shape: [1, 3, num_video_frames//4, H, W]  (用于可视化保存)

# Action 输出 (直接使用去噪后的 latent)
predicted_actions = action_latent.float()  # [1, 16, 14]
```

**注意**: Action 输出**不经过反归一化**。模型直接在原始关节角空间训练和预测。归一化只用于 state 输入的条件化 (`current_state_norm`)，但推理输出的 action 是原始尺度。

---

## 9. Action 输出与反归一化

### 9.1 Action 格式

```python
predicted_actions.shape = [1, 16, 14]
# 维度说明:
#   B=1:  batch size
#   16:   action_chunk_size = num_video_frames(8) × video_action_freq_ratio(2)
#   14:   action_dim = [left_arm_j1..j6, left_gripper, right_arm_j1..j6, right_gripper]
```

### 9.2 Action 内容 (14 维)

| Index | 含义 | 范围 (训练数据统计) |
|-------|------|---------------------|
| 0 | 左臂关节1 | [-1.42, 0.52] |
| 1 | 左臂关节2 | [-0.005, 3.46] |
| 2 | 左臂关节3 | [-0.19, 3.69] |
| 3 | 左臂关节4 | [-1.96, 1.79] |
| 4 | 左臂关节5 | [-1.68, 1.86] |
| 5 | 左臂关节6 | [-3.97, 3.99] |
| 6 | 左夹爪 | [0.0, 1.0] |
| 7 | 右臂关节1 | [-2.95, 1.45] |
| 8 | 右臂关节2 | [-0.63, 3.64] |
| 9 | 右臂关节3 | [-0.08, 3.99] |
| 10 | 右臂关节4 | [-2.04, 1.96] |
| 11 | 右臂关节5 | [-2.13, 1.53] |
| 12 | 右臂关节6 | [-4.0, 3.96] |
| 13 | 右夹爪 | [0.0, 1.0] |

### 9.3 从模型输出到执行

```python
# deploy_policy.py L276-280
actions_real = predicted_actions.squeeze(0).cpu().numpy()  # [16, 14]
model.prev_action = actions_real[-1].copy()
model.action_cache.extend(actions_real)
return actions_real  # List[np.ndarray], 每个 shape (14,)
```

```python
# deploy_policy.py L426-437 (eval 函数)
actions = model.get_action()  # [16, 14]
for action in actions:  # 逐个执行 16 个 action
    TASK_ENV.take_action(action, action_type='qpos')
```

---

## 10. Action 执行 (take_action)

**文件**: `envs/_base_task.py` → `take_action()` (L1479-1667)

### 10.1 Action 解析

```python
# action shape: (14,)
# 解析为:
left_arm_actions  = action[:6]     # 左臂 6 关节目标角度
left_gripper      = action[6]      # 左夹爪目标值
right_arm_actions = action[7:13]   # 右臂 6 关节目标角度
right_gripper     = action[13]     # 右夹爪目标值
```

### 10.2 TOPP 轨迹规划

每个 action 不是直接 set 关节角，而是经过 **TOPP (Time-Optimal Path Parameterization)** 规划:

```
当前关节角 → 目标关节角
       ↓
   TOPP 规划 (toppra)
       ↓
平滑轨迹: N 步 (position + velocity)
       ↓
逐步执行 scene.step()
```

- 使用 `mplib_planner.TOPP()` 做时间最优路径参数化
- 采样率: 1/250 (250Hz 控制频率)
- 如果 TOPP 失败，fallback 到 50 步线性插值
- 夹爪值在 TOPP 步数内线性插值

### 10.3 控制循环

```python
while now_left_id < left_n_step or now_right_id < right_n_step:
    # 按比例交替执行左右臂 (同步双臂)
    if left_arm_turn:
        robot.set_arm_joints(left_pos[i], left_vel[i], "left")
        robot.set_gripper(left_gripper[i], "left")
    if right_arm_turn:
        robot.set_arm_joints(right_pos[i], right_vel[i], "right")
        robot.set_gripper(right_gripper[i], "right")
    
    scene.step()  # SAPIEN 物理仿真步
    
    if check_success():
        eval_success = True
        return
```

### 10.4 成功判定

- 每个 `scene.step()` 后检查 `check_success()`
- 各 task 自定义成功条件 (如瓶子在垃圾桶内)
- 每个 episode 有步数上限 (`_eval_step_limit.yml`)，如 `put_bottles_dustbin: 1700`

---

## 11. 评测循环总结

### 单次 eval 调用的数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ SAPIEN 仿真环境                                                  │
│  ├─ get_obs() → 3相机RGB + 14维关节角                             │
│  └─ take_action(action_14d, 'qpos') → TOPP规划 → 物理仿真        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ MotusPolicy.update_obs(observation)                              │
│  ├─ 三相机拼接 → resize_with_padding → [1, 3, 384, 320]         │
│  └─ joint_action.vector → [1, 14] + normalize                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ MotusPolicy.get_action()                                         │
│  ├─ T5 编码 instruction → [1, 512, 4096]                         │
│  ├─ VLM 编码 (instruction + image) → vlm_inputs                  │
│  └─ Motus.inference_step():                                      │
│      ├─ VAE encode 条件帧 → [1, 48, 1, 12, 10]                   │
│      ├─ 初始化 video_latent [1,48,3,12,10] + action_latent [1,16,14] │
│      ├─ VLM → UndExpert features [1, seq, 512]                   │
│      ├─ T5 → text_embedding [1, 512, 3072]                       │
│      ├─ 10 步 Euler 去噪:                                        │
│      │   └─ 30 层 Trimodal MoT Transformer                       │
│      │       ├─ Joint Self-Attn (Video+Action+Und in WAN space)   │
│      │       ├─ Cross-Attn (Video × T5)                           │
│      │       └─ Independent FFNs × 3                              │
│      ├─ VAE decode video (可视化)                                 │
│      └─ action_latent → [1, 16, 14] (原始关节角)                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 执行 16 个 actions                                               │
│  for action in actions[16]:                                      │
│      TASK_ENV.take_action(action, 'qpos')                        │
│      → TOPP 规划 + N 步物理仿真 + 成功检测                         │
└─────────────────────────────────────────────────────────────────┘
```

### 每个 episode 的步骤计数

- 模型每次推理输出 **16 个 action** (action_chunk_size)
- 每个 action 经 TOPP 展开为 ~50-250 个物理仿真步
- `step_lim` 计数的是 `take_action` 调用次数 (即模型输出的 action 个数)
- 例如 `put_bottles_dustbin: step_lim=1700`，即最多执行 1700 个 action
- 每次 `get_action()` 产生 16 个 action，所以最多调用 ~106 次模型推理

### obs_cache 说明

- `obs_cache = deque(maxlen=1)`: 只保留最新一帧
- 每次 `get_action()` 都使用**当前最新的单帧**作为条件
- 没有历史帧堆叠，每次推理独立

---

## 12. 关键配置参数

### robotwin.yml

| 参数 | 值 | 说明 |
|------|---|------|
| `action_dim` | 14 | 机器人动作维度 (6关节+1夹爪) × 2 |
| `state_dim` | 14 | 机器人状态维度 (同 action) |
| `num_video_frames` | 8 | 预测视频帧数 |
| `video_height` | 384 | 输入图像高度 |
| `video_width` | 320 | 输入图像宽度 |
| `global_downsample_rate` | 3 | 全局降采样率 |
| `video_action_freq_ratio` | 2 | 视频:动作频率比 |
| `action_chunk_size` | 16 | = 8 × 2, 动作序列长度 |
| `num_inference_timesteps` | 10 | 去噪步数 |
| `action_expert.hidden_size` | 1024 | Action Expert 隐藏维度 |
| `und_expert.hidden_size` | 512 | Understanding Expert 隐藏维度 |

### deploy_policy.yml

| 参数 | 值 | 说明 |
|------|---|------|
| `instruction_type` | unseen | 使用未见过的指令变体 |
| `model_config` | robotwin.yml | 模型配置文件 |

### demo_randomized.yml (任务环境配置)

| 参数 | 值 | 说明 |
|------|---|------|
| `random_background` | true | 随机背景纹理 |
| `cluttered_table` | true | 桌面杂物 |
| `random_light` | true | 随机光照 |
| `random_table_height` | 0.03 | 桌面高度随机偏移 |
| `head_camera_type` | D435 | 头部相机型号 |
| `episode_num` | 50 | 每轮 episode 数 |

---

## 附录: 文件索引

| 文件 | 作用 |
|------|------|
| `policy/Motus/debug_single.sh` | 单 GPU 调试启动脚本 |
| `script/eval_policy.py` | 评测主循环 |
| `policy/Motus/deploy_policy.py` | MotusPolicy 封装 (obs处理, 推理, action执行) |
| `policy/Motus/deploy_policy.yml` | 部署配置 |
| `policy/Motus/utils/robotwin.yml` | 模型架构配置 |
| `policy/Motus/utils/stat.json` | 动作归一化统计量 |
| `policy/Motus/utils/image_utils.py` | 图像预处理工具 |
| `policy/Motus/models/motus.py` | Motus 主模型 (三模态 UniDiffuser) |
| `policy/Motus/models/action_expert.py` | Action Expert 模块 |
| `policy/Motus/models/und_expert.py` | Understanding Expert 模块 |
| `policy/Motus/models/wan_model.py` | WAN 视频模型封装 |
| `envs/_base_task.py` | 基础任务环境 (get_obs, take_action) |
| `task_config/demo_randomized.yml` | 任务环境配置 |
| `task_config/_eval_step_limit.yml` | 各任务步数上限 |
