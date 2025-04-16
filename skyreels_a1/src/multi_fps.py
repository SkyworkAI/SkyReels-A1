import math
import torch
import numpy as np
from PIL import Image
import subprocess
from .frame_interpolation import batch_images_interpolation_tool

def multi_fps_tool(frames, frame_inter_model, target_fps, original_fps=12):
    frames_np = np.array([np.array(frame) for frame in frames])
    
    interpolation_factor = target_fps / original_fps
    inter_frames = math.ceil(interpolation_factor) - 1
    frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
    
    video = batch_images_interpolation_tool(frames_tensor, frame_inter_model, inter_frames=inter_frames)
    video = video.squeeze(0)
    video = video.permute(1, 2, 3, 0)
    video = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    out_frames = [Image.fromarray(frame) for frame in video]

    if not interpolation_factor.is_integer():
        print(f"Warning: target fps {target_fps} is not mulitple of 12, which may cause unstable video rate.")
        out_frames = adjust_video_fps(out_frames, target_fps, int(target_fps//12+1)*12)
    
    return out_frames

def adjust_video_fps(frames, target_fps, fps):
    video_length = len(frames)

    duration = video_length / fps 
    target_times = np.arange(0, duration, 1/target_fps)
    frame_indices = (target_times * fps).astype(np.int32)

    frame_indices = frame_indices[frame_indices < video_length]
    new_frames = []
    for idx in frame_indices:
        new_frames.append(frames[idx])

    return new_frames