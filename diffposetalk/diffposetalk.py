import math
import tempfile
import warnings
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pydantic import BaseModel

from .diff_talking_head import DiffTalkingHead
from .utils import NullableArgs, coef_dict_to_vertices, get_coef_dict
from .utils.media import combine_video_and_audio, convert_video, reencode_audio

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')

class DiffPoseTalkConfig(BaseModel):
    no_context_audio_feat: bool = False
    model_path: str = "pretrained_models/diffposetalk/iter_0110000.pt" # DPT/head-SA-hubert-WM
    coef_stats: str = "pretrained_models/diffposetalk/stats_train.npz"
    style_path: str = "pretrained_models/diffposetalk/style/L4H4-T0.1-BS32/iter_0034000/normal.npy"
    dynamic_threshold_ratio: float = 0.99
    dynamic_threshold_min: float = 1.0
    dynamic_threshold_max: float = 4.0
    scale_audio: float = 1.15
    scale_style: float = 3.0

class DiffPoseTalk:
    def __init__(self, config: DiffPoseTalkConfig = DiffPoseTalkConfig(), device="cuda"):
        self.cfg = config
        self.device = device

        self.no_context_audio_feat = self.cfg.no_context_audio_feat
        model_data = torch.load(self.cfg.model_path, map_location=self.device)

        self.model_args = NullableArgs(model_data['args'])
        self.model = DiffTalkingHead(self.model_args, self.device)
        model_data['model'].pop('denoising_net.TE.pe')
        self.model.load_state_dict(model_data['model'], strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.use_indicator = self.model_args.use_indicator
        self.rot_repr = self.model_args.rot_repr
        self.predict_head_pose = not self.model_args.no_head_pose
        if self.model.use_style:
            style_dir = Path(self.model_args.style_enc_ckpt)
            style_dir = Path(*style_dir.with_suffix('').parts[-3::2])
            self.style_dir = style_dir

        # sequence
        self.n_motions = self.model_args.n_motions
        self.n_prev_motions = self.model_args.n_prev_motions
        self.fps = self.model_args.fps
        self.audio_unit = 16000. / self.fps  # num of samples per frame
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.pad_mode = self.model_args.pad_mode

        self.coef_stats = dict(np.load(self.cfg.coef_stats))
        self.coef_stats = {k: torch.from_numpy(v).to(self.device) for k, v in self.coef_stats.items()}

        if self.cfg.dynamic_threshold_ratio > 0:
            self.dynamic_threshold = (self.cfg.dynamic_threshold_ratio, self.cfg.dynamic_threshold_min,
                                      self.cfg.dynamic_threshold_max)
        else:
            self.dynamic_threshold = None


    def infer_from_file(self, audio_path, shape_coef):
        n_repetitions = 1
        cfg_mode = None
        cfg_cond = self.model.guiding_conditions
        cfg_scale = []
        for cond in cfg_cond:
            if cond == 'audio':
                cfg_scale.append(self.cfg.scale_audio)
            elif cond == 'style':
                cfg_scale.append(self.cfg.scale_style)

        coef_dict = self.infer_coeffs(audio_path, shape_coef, self.cfg.style_path, n_repetitions,
                                      cfg_mode, cfg_cond, cfg_scale, include_shape=True)
        return coef_dict

    @torch.no_grad()
    def infer_coeffs(self, audio, shape_coef, style_feat=None, n_repetitions=1,
                     cfg_mode=None, cfg_cond=None, cfg_scale=1.15, include_shape=False):
        # Returns dict[str, (n_repetitions, L, *)]
        # Step 1: Preprocessing
        # Preprocess audio
        if isinstance(audio, (str, Path)):
            audio, _ = librosa.load(audio, sr=16000, mono=True)
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).to(self.device)
        assert audio.ndim == 1, 'Audio must be 1D tensor.'
        audio_mean, audio_std = torch.mean(audio), torch.std(audio)
        audio = (audio - audio_mean) / (audio_std + 1e-5)

        # Preprocess shape coefficient
        if isinstance(shape_coef, (str, Path)):
            shape_coef = np.load(shape_coef)
            if not isinstance(shape_coef, np.ndarray):
                shape_coef = shape_coef['shape']
        if isinstance(shape_coef, np.ndarray):
            shape_coef = torch.from_numpy(shape_coef).float().to(self.device)
        assert shape_coef.ndim <= 2, 'Shape coefficient must be 1D or 2D tensor.'
        if shape_coef.ndim > 1:
            # use the first frame as the shape coefficient
            shape_coef = shape_coef[0]
        original_shape_coef = shape_coef.clone()
        if self.coef_stats is not None:
            shape_coef = (shape_coef - self.coef_stats['shape_mean']) / self.coef_stats['shape_std']
        shape_coef = shape_coef.unsqueeze(0).expand(n_repetitions, -1)

        # Preprocess style feature if given
        if style_feat is not None:
            assert self.model.use_style
            if isinstance(style_feat, (str, Path)):
                style_feat = Path(style_feat)
                if not style_feat.exists() and not style_feat.is_absolute():
                    style_feat = style_feat.parent / self.style_dir / style_feat.name
                style_feat = np.load(style_feat)
                if not isinstance(style_feat, np.ndarray):
                    style_feat = style_feat['style']
            if isinstance(style_feat, np.ndarray):
                style_feat = torch.from_numpy(style_feat).float().to(self.device)
            assert style_feat.ndim == 1, 'Style feature must be 1D tensor.'
            style_feat = style_feat.unsqueeze(0).expand(n_repetitions, -1)

        # Step 2: Predict motion coef
        # divide into synthesize units and do synthesize
        clip_len = int(len(audio) / 16000 * self.fps)
        stride = self.n_motions
        if clip_len <= self.n_motions:
            n_subdivision = 1
        else:
            n_subdivision = math.ceil(clip_len / stride)

        # Prepare audio input
        n_padding_audio_samples = self.n_audio_samples * n_subdivision - len(audio)
        n_padding_frames = math.ceil(n_padding_audio_samples / self.audio_unit)
        if n_padding_audio_samples > 0:
            if self.pad_mode == 'zero':
                padding_value = 0
            elif self.pad_mode == 'replicate':
                padding_value = audio[-1]
            else:
                raise ValueError(f'Unknown pad mode: {self.pad_mode}')
            audio = F.pad(audio, (0, n_padding_audio_samples), value=padding_value)

        if not self.no_context_audio_feat:
            audio_feat = self.model.extract_audio_feature(audio.unsqueeze(0), self.n_motions * n_subdivision)

        # Generate `self.n_motions` new frames at one time, and use the last `self.n_prev_motions` frames
        # from the previous generation as the initial motion condition
        coef_list = []
        for i in range(0, n_subdivision):
            start_idx = i * stride
            end_idx = start_idx + self.n_motions
            indicator = torch.ones((n_repetitions, self.n_motions)).to(self.device) if self.use_indicator else None
            if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
                indicator[:, -n_padding_frames:] = 0
            if not self.no_context_audio_feat:
                audio_in = audio_feat[:, start_idx:end_idx].expand(n_repetitions, -1, -1)
            else:
                audio_in = audio[round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)].unsqueeze(0)

            # generate motion coefficients
            if i == 0:
                # -> (N, L, d_motion=n_code_per_frame * code_dim)
                motion_feat, noise, prev_audio_feat = self.model.sample(audio_in, shape_coef, style_feat,
                                                                        indicator=indicator, cfg_mode=cfg_mode,
                                                                        cfg_cond=cfg_cond, cfg_scale=cfg_scale,
                                                                        dynamic_threshold=self.dynamic_threshold)
            else:
                motion_feat, noise, prev_audio_feat = self.model.sample(audio_in, shape_coef, style_feat,
                                                                        prev_motion_feat, prev_audio_feat, noise,
                                                                        indicator=indicator, cfg_mode=cfg_mode,
                                                                        cfg_cond=cfg_cond, cfg_scale=cfg_scale,
                                                                        dynamic_threshold=self.dynamic_threshold)
            prev_motion_feat = motion_feat[:, -self.n_prev_motions:].clone()
            prev_audio_feat = prev_audio_feat[:, -self.n_prev_motions:]

            motion_coef = motion_feat
            if i == n_subdivision - 1 and n_padding_frames > 0:
                motion_coef = motion_coef[:, :-n_padding_frames]  # delete padded frames
            coef_list.append(motion_coef)

        motion_coef = torch.cat(coef_list, dim=1)

        # Step 3: restore to coef dict
        coef_dict = get_coef_dict(motion_coef, None, self.coef_stats, self.predict_head_pose, self.rot_repr)
        if include_shape:
            coef_dict['shape'] = original_shape_coef[None, None].expand(n_repetitions, motion_coef.shape[1], -1)
        return self.coef_to_a1_format(coef_dict)

    def coef_to_a1_format(self, coef_dict):
        n_frames = coef_dict['exp'].shape[1]
        new_coef_dict = []
        for i in range(n_frames):

            new_coef_dict.append({
                "expression_params": coef_dict["exp"][0, i:i+1],
                "jaw_params": coef_dict["pose"][0, i:i+1, 3:],
                "eye_pose_params": torch.zeros(1, 6).type_as(coef_dict["pose"]),
                "pose_params": coef_dict["pose"][0, i:i+1, :3],
                "eyelid_params": None
            })
        return new_coef_dict
        




    @staticmethod
    def _pad_coef(coef, n_frames, elem_ndim=1):
        if coef.ndim == elem_ndim:
            coef = coef[None]
        elem_shape = coef.shape[1:]
        if coef.shape[0] >= n_frames:
            new_coef = coef[:n_frames]
        else:
            # repeat the last coef frame
            new_coef = torch.cat([coef, coef[[-1]].expand(n_frames - coef.shape[0], *elem_shape)], dim=0)
        return new_coef  # (n_frames, *elem_shape)

