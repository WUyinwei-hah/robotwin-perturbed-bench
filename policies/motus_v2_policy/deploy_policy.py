# MotusV2 Policy for RoboTwin

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import logging
import subprocess
from typing import List, Dict, Any, Optional
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add model paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

from models.motus_v2 import MotusV2, MotusV2Config

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel
from utils.image_utils import resize_with_padding

logger = logging.getLogger(__name__)


class MotusV2Policy:
    """
    MotusV2 Policy wrapper for RoboTwin evaluation.
    Uses 9 condition frames + 16 condition actions for inference.
    """

    def __init__(self, checkpoint_path: str, config_path: str, wan_path: str, vlm_path: str,
                 device: str = "cuda", log_dir: Optional[str] = None, task_name: Optional[str] = None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path

        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)

        # Initialize model
        self.model = self._load_model()

        # Initialize T5 encoder for language embeddings
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(self.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.wan_path, 'google', 'umt5-xxl'),
        )

        # Initialize VLM processor (tokenization only, weights from checkpoint)
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)

        # V2 buffers
        self.video_action_freq_ratio = self.config_dict['common']['video_action_freq_ratio']
        self.frame_buffer = deque(maxlen=20)    # real env observations [1, C, H, W]
        self.action_history = deque(maxlen=32)  # raw executed actions [14]
        self.action_cache = deque()

        # Model state
        self.initial_state = None    # raw qpos at episode start
        self.current_state = None
        self.is_first_step = True
        self.current_instruction = None

        # Predicted video accumulation
        self.pred_video_frames = []
        self.pred_video_save_dir = None
        self.pred_video_episode_num = 0

        # Image saving
        self.save_images = True
        base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        task_dir_name = task_name or os.environ.get('TASK_NAME') or "default_task"
        self.save_dir = Path(base_log_dir) / "images" / task_dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0
        self.step_count = 0

        logger.info("MotusV2 Policy initialized successfully")

    def set_instruction(self, instruction: str):
        """Set the current instruction for the policy."""
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def _load_model(self) -> MotusV2:
        """Load the MotusV2 model without pretrained backbones, then load checkpoint."""
        logger.info(f"Initializing MotusV2 model from config (no pretrained backbones)")

        config = self._create_model_config()

        model = MotusV2(config)
        model = model.to(self.device)

        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info("Model checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

        model.eval()
        return model

    def _create_model_config(self) -> MotusV2Config:
        """Create V2 model configuration from yaml config — inference mode."""
        common = self.config_dict['common']
        model_cfg = self.config_dict['model']

        vae_path = os.path.join(self.wan_path, "Wan2.2_VAE.pth")

        hidden_size = model_cfg['action_expert']['hidden_size']
        ffn_multiplier = model_cfg['action_expert']['ffn_dim_multiplier']

        config = MotusV2Config(
            # Paths (no pretrained weights loaded — full checkpoint loaded later)
            wan_checkpoint_path=self.wan_path,
            vae_path=vae_path,
            wan_config_path=self.wan_path,
            video_precision='bfloat16',
            vlm_checkpoint_path=self.vlm_path,

            # Understanding expert config
            und_expert_hidden_size=model_cfg['und_expert']['hidden_size'],
            und_expert_ffn_dim_multiplier=model_cfg['und_expert']['ffn_dim_multiplier'],
            und_expert_norm_eps=model_cfg['und_expert']['norm_eps'],
            und_layers_to_extract=None,
            vlm_adapter_input_dim=model_cfg['und_expert']['vlm']['input_dim'],
            vlm_adapter_projector_type=model_cfg['und_expert']['vlm']['projector_type'],

            # Model architecture
            num_layers=30,
            action_state_dim=common['state_dim'],
            action_dim=common['action_dim'],
            action_expert_dim=hidden_size,
            action_expert_ffn_dim_multiplier=ffn_multiplier,
            action_expert_norm_eps=1e-6,

            # V2 frame settings
            num_condition_frames=common.get('num_condition_frames', 9),
            num_predicted_frames=common.get('num_predicted_frames', 16),

            # Training config (needed for derived params)
            global_downsample_rate=common['global_downsample_rate'],
            video_action_freq_ratio=common['video_action_freq_ratio'],
            num_video_frames=common['num_video_frames'],
            video_loss_weight=1.0,
            action_loss_weight=1.0,

            # Inference config
            batch_size=1,
            video_height=common['video_height'],
            video_width=common['video_width'],

            # Don't load pretrained backbones — full model from checkpoint
            load_pretrained_backbones=False,
            training_mode='finetune',
        )

        return config

    def update_obs(self, observation: Dict[str, Any]):
        """Update observation cache with new observation."""
        # Extract visual observations
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                head_img = obs_data['head_camera']['rgb']
                left_img = obs_data['left_camera']['rgb']
                right_img = obs_data['right_camera']['rgb']

                left_img_resized = cv2.resize(left_img, (160, 120))
                right_img_resized = cv2.resize(right_img, (160, 120))
                bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
                image = np.concatenate([head_img, bottom_row], axis=0)
            else:
                raise ValueError("Missing camera data")
        elif 'head_camera' in observation:
            image = observation['head_camera']
        elif 'image' in observation:
            image = observation['image']
        else:
            raise ValueError("No visual observation found")

        target_size = (self.config_dict['common']['video_height'],
                      self.config_dict['common']['video_width'])

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image

        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)

        self.frame_buffer.append(image_tensor.to(self.device))

        # Extract robot state
        state = observation['joint_action']['vector']
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float()
        else:
            state_tensor = state.float() if state.dim() == 1 else state.float().squeeze(0)

        self.current_state = state_tensor.to(self.device)

        # Store initial state on first observation
        if self.initial_state is None:
            self.initial_state = self.current_state.clone()

    def collect_intermediate_frame(self, observation: Dict[str, Any]):
        """Collect an intermediate observation frame during action execution."""
        try:
            if 'observation' in observation:
                obs_data = observation['observation']
                if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                    head_img = obs_data['head_camera']['rgb']
                    left_img = obs_data['left_camera']['rgb']
                    right_img = obs_data['right_camera']['rgb']
                    left_img_resized = cv2.resize(left_img, (160, 120))
                    right_img_resized = cv2.resize(right_img, (160, 120))
                    bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
                    image = np.concatenate([head_img, bottom_row], axis=0)
                else:
                    return
            elif 'head_camera' in observation:
                image = observation['head_camera']
            elif 'image' in observation:
                image = observation['image']
            else:
                return

            target_size = (self.config_dict['common']['video_height'],
                          self.config_dict['common']['video_width'])

            if isinstance(image, np.ndarray):
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            else:
                image_tensor = image

            if image_tensor.shape[-2:] != target_size:
                image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                resized_np = resize_with_padding(image_np, target_size)
                if resized_np.dtype == np.uint8:
                    resized_np = resized_np.astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)

            self.frame_buffer.append(image_tensor.to(self.device))
        except Exception as e:
            logger.warning(f"Failed to collect intermediate frame: {e}")

    def get_action(self) -> np.ndarray:
        """Get action predictions from the model.

        Returns:
            actions: [32, 14] numpy array of raw qpos actions
        """
        if len(self.frame_buffer) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")

        # ===================== BUILD CONDITION FRAMES =====================
        num_cond_frames = self.config_dict['common'].get('num_condition_frames', 9)

        if self.is_first_step:
            # Step 0: repeat first frame × 9 (matches training is_first_frame=True)
            first_frame = self.frame_buffer[0]  # [1, C, H, W]
            cond_frames_list = [first_frame] * num_cond_frames
        else:
            # Subsequent steps: use last 9 real observations
            frames_available = list(self.frame_buffer)
            if len(frames_available) >= num_cond_frames:
                cond_frames_list = frames_available[-num_cond_frames:]
            else:
                # Pad with first available frame
                pad_count = num_cond_frames - len(frames_available)
                cond_frames_list = [frames_available[0]] * pad_count + frames_available

        # Stack: [9, 1, C, H, W] → [1, 9, C, H, W]
        condition_frames = torch.cat(cond_frames_list, dim=0).unsqueeze(0)  # [1, 9, C, H, W]

        # ===================== BUILD CONDITION ACTIONS =====================
        num_cond_actions = 16  # (num_condition_frames - 1) * video_action_freq_ratio

        if self.is_first_step:
            # Step 0: repeat initial qpos × 16 (matches training is_first_frame=True)
            cond_action = self.initial_state.unsqueeze(0)  # [1, 14]
            condition_actions = cond_action.expand(num_cond_actions, -1).unsqueeze(0)  # [1, 16, 14]
        else:
            # Subsequent steps: use last 16 executed actions
            actions_available = list(self.action_history)
            if len(actions_available) >= num_cond_actions:
                cond_actions_list = actions_available[-num_cond_actions:]
            else:
                # Pad with first available action (or initial state)
                pad_val = actions_available[0] if actions_available else self.initial_state
                pad_count = num_cond_actions - len(actions_available)
                cond_actions_list = [pad_val] * pad_count + actions_available
            condition_actions = torch.stack(cond_actions_list).unsqueeze(0).to(self.device)  # [1, 16, 14]

        # ===================== ENCODE INSTRUCTION =====================
        scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                        "The aloha robot is currently performing the following task: ")
        instruction = f"{scene_prefix}{self.current_instruction}"
        t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        # ===================== VLM INPUTS =====================
        # Use the latest observation for VLM
        current_frame = self.frame_buffer[-1]
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(instruction, first_frame_pil)

        # ===================== INFERENCE =====================
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                condition_frames=condition_frames,
                condition_actions=condition_actions,
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs],
            )

        # ===================== SAVE FRAME GRID =====================
        if predicted_frames is not None:
            try:
                # predicted_frames: [1, 16, C, H, W]
                pf_viz = predicted_frames.squeeze(0)  # [16, C, H, W]
                cond_frame_viz = current_frame.squeeze(0)  # [C, H, W]
                self._save_frame_grid(cond_frame_viz, pf_viz)
                self.step_count += 1
            except Exception as e:
                logger.warning(f"Failed to save frame grid: {e}")

        # Collect predicted frames for video
        if predicted_frames is not None and self.pred_video_save_dir is not None:
            try:
                pf = predicted_frames.squeeze(0)  # [16, C, H, W]
                num_pred_frames = pf.shape[0]
                num_actions = predicted_actions.shape[1]
                reps_per_frame = max(1, num_actions // num_pred_frames)
                for i in range(num_pred_frames):
                    frame_np = (pf[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                    for _ in range(reps_per_frame):
                        self.pred_video_frames.append(frame_np)
            except Exception as e:
                logger.warning(f"Failed to collect predicted frame: {e}")

        # ===================== RETURN ACTIONS =====================
        actions_np = predicted_actions.squeeze(0).cpu().numpy()  # [32, 14]
        return actions_np

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        """Convert [C, H, W] tensor to PIL Image."""
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Build VLM inputs."""
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': instruction},
                    {'type': 'image', 'image': image},
                ]
            }
        ]
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device),
            'pixel_values': encoded['pixel_values'].to(self.device),
        }
        if 'image_grid_thw' in encoded:
            vlm_inputs['image_grid_thw'] = encoded['image_grid_thw'].to(self.device)
        return vlm_inputs

    def _create_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:
        """Create horizontal grid of condition + predicted frames."""
        def tensor_to_numpy(tensor):
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            tensor = tensor.detach().cpu().float()
            tensor = torch.clamp(tensor, 0, 1)
            return (tensor.numpy() * 255).astype(np.uint8)

        condition_np = tensor_to_numpy(condition_frame)
        predicted_np = []
        num_pred_frames = predicted_frames.shape[0]
        for i in range(min(num_pred_frames, 4)):
            frame_np = tensor_to_numpy(predicted_frames[i])
            predicted_np.append(frame_np)

        while len(predicted_np) < 4:
            predicted_np.append(predicted_np[-1] if predicted_np else condition_np)

        all_frames = [condition_np] + predicted_np[:4]
        grid_image = np.concatenate(all_frames, axis=1)

        return Image.fromarray(grid_image)

    def start_pred_video(self, save_dir: str, episode_num: int):
        """Start accumulating predicted frames for a new episode."""
        self.pred_video_save_dir = save_dir
        self.pred_video_episode_num = episode_num
        self.pred_video_frames = []
        logger.info(f"Started predicted video collection for episode {episode_num}")

    def flush_pred_video(self):
        """Save accumulated predicted frames as a video."""
        if self.pred_video_frames and self.pred_video_save_dir:
            try:
                h, w = self.pred_video_frames[0].shape[:2]
                video_path = os.path.join(
                    str(self.pred_video_save_dir),
                    f"episode{self.pred_video_episode_num}_pred.mp4"
                )
                proc = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error",
                     "-f", "rawvideo", "-pixel_format", "rgb24",
                     "-video_size", f"{w}x{h}",
                     "-framerate", "10",
                     "-i", "-",
                     "-pix_fmt", "yuv420p",
                     "-vcodec", "libx264",
                     "-crf", "23",
                     video_path],
                    stdin=subprocess.PIPE
                )
                for frame in self.pred_video_frames:
                    proc.stdin.write(frame.tobytes())
                proc.stdin.close()
                proc.wait()
                logger.info(f"Saved predicted video ({len(self.pred_video_frames)} frames) to {video_path}")
            except Exception as e:
                logger.warning(f"Failed to save predicted video: {e}")
        self.pred_video_frames = []
        self.pred_video_save_dir = None

    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        """Save frame grid to disk."""
        if not self.save_images:
            return
        try:
            grid_image = self._create_frame_grid(condition_frame, predicted_frames)
            filename = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.png"
            save_path = self.save_dir / filename
            grid_image.save(save_path)
            logger.info(f"Saved frame grid to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save frame grid: {e}")


def encode_obs(observation):
    """Post-Process Observation"""
    return observation


def get_model(usr_args):
    """
    Initialize MotusV2 model.

    Args:
        usr_args: Arguments from eval script (must include wan_path and vlm_path)
    """
    checkpoint_path = usr_args.get('ckpt_setting')
    wan_path = usr_args.get('wan_path')
    vlm_path = usr_args.get('vlm_path')

    if not wan_path:
        raise ValueError("wan_path not provided in usr_args")
    if not vlm_path:
        raise ValueError("vlm_path not provided in usr_args")

    policy_dir = Path(__file__).parent
    config_path = policy_dir / "utils" / "robotwin_v2.yml"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = MotusV2Policy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        config_path=str(config_path),
        device=device,
        log_dir=usr_args.get('log_dir'),
        task_name=usr_args.get('task_name'),
    )

    return policy


def eval(TASK_ENV, model, observation):
    """Evaluation function with intermediate observation collection for V2."""
    obs = encode_obs(observation)

    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)

    actions = model.get_action()  # [32, 14] raw qpos

    freq_ratio = model.video_action_freq_ratio  # 2

    for i, action in enumerate(actions):
        TASK_ENV.take_action(action, action_type='qpos')
        # Store executed action in history (raw qpos)
        model.action_history.append(torch.from_numpy(action).float().to(model.device))
        model.action_cache.append(action)

        # Collect intermediate observation every freq_ratio actions (video rate)
        if (i + 1) % freq_ratio == 0 and i < len(actions) - 1:
            try:
                inter_obs = TASK_ENV.get_obs()
                model.collect_intermediate_frame(inter_obs)
            except Exception as e:
                logger.warning(f"Failed to get intermediate obs: {e}")

    model.is_first_step = False


def reset_model(model):
    """Reset model cache at episode start."""
    model.frame_buffer.clear()
    model.action_history.clear()
    model.action_cache.clear()
    model.initial_state = None
    model.current_state = None
    model.is_first_step = True
    model.episode_count += 1
    model.step_count = 0
    logger.info(f"Model reset completed for episode {model.episode_count}")
