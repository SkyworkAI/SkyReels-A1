import torch
import os
import numpy as np
from PIL import Image
import glob
import insightface 
import cv2
import subprocess
import argparse
import time
from decord import VideoReader
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from insightface.app import FaceAnalysis 

from diffusers.models import AutoencoderKLCogVideoX
from diffusers.utils import export_to_video, load_image
from transformers import AutoModelForDepthEstimation, AutoProcessor, SiglipImageProcessor, SiglipVisionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from skyreels_a1.models.transformer3d import CogVideoXTransformer3DModel
from skyreels_a1.skyreels_a1_i2v_long_pipeline import SkyReelsA1ImagePoseToVideoPipeline
from skyreels_a1.pre_process_lmk3d import FaceAnimationProcessor
from skyreels_a1.src.media_pipe.mp_utils import LMKExtractor
from skyreels_a1.src.media_pipe.draw_util_2d import FaceMeshVisualizer2d
from skyreels_a1.src.frame_interpolation import init_frame_interpolation_model, batch_images_interpolation_tool

from diffusers.video_processor import VideoProcessor


def crop_and_resize(image, height, width):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image = image[:, left: left+croped_width]
        image = Image.fromarray(image).resize((width, height))
    else:
        pad = int((((width / height) * image_height) - image_width) / 2.)
        padded_image = np.zeros((image_height, image_width + pad * 2, 3), dtype=np.uint8)
        padded_image[:, pad:pad+image_width] = image
        image = Image.fromarray(padded_image).resize((width, height))
    return image


def write_mp4(video_path, samples, fps=12, audio_bitrate="192k"):
    clip = ImageSequenceClip(samples, fps=fps)
    clip.write_videofile(video_path, audio_codec="aac", audio_bitrate=audio_bitrate, 
                         ffmpeg_params=["-crf", "18", "-preset", "slow"])


def parse_video(driving_video_path, max_frame_num=10000, frame_num_per_batch=49, target_fps=12):
    vr = VideoReader(driving_video_path)
    fps = vr.get_avg_fps()
    video_length = len(vr)

    duration = video_length / fps 
    target_times = np.arange(0, duration, 1/target_fps)  # 12 fps
    frame_indices = (target_times * fps).astype(np.int32)

    frame_indices = frame_indices[frame_indices < video_length]
    control_frames = vr.get_batch(frame_indices).asnumpy()[:max_frame_num]
    
    return control_frames


def exec_cmd(cmd):
            return subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def add_audio_to_video(silent_video_path, audio_video_path, output_video_path):
    cmd = [
        'ffmpeg',
        '-y',
        '-i', f'"{silent_video_path}"',
        '-i', f'"{audio_video_path}"',
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-shortest',
        f'"{output_video_path}"'
    ]

    try:
        exec_cmd(' '.join(cmd))
        print(f"Video with audio generated successfully: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


def smooth_video_transition(frames1, frames2, smooth_frame_num, frame_inter_model, inter_frames=2):

    frames1_np = np.array([np.array(frame) for frame in frames1])
    frames2_np = np.array([np.array(frame) for frame in frames2])
    frames1_tensor = torch.from_numpy(frames1_np).permute(3,0,1,2).unsqueeze(0) / 255.0
    frames2_tensor = torch.from_numpy(frames2_np).permute(3,0,1,2).unsqueeze(0) / 255.0
    video = torch.cat([frames1_tensor[:,:,-smooth_frame_num:], frames2_tensor[:,:,:smooth_frame_num]], dim=2)

    video = batch_images_interpolation_tool(video, frame_inter_model, inter_frames=inter_frames)

    index = [1, 4, 5, 8] if inter_frames == 2 else [2, 5, 7, 10]
    video = video[:, :, index]
    video = video.squeeze(0)
    video = video.permute(1, 2, 3, 0)
    video = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    mid_frames = [Image.fromarray(frame) for frame in video]

    out_frames = frames1[:-smooth_frame_num] + mid_frames + frames2[smooth_frame_num:]

    return out_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and image for face animation.")
    parser.add_argument('--image_path', type=str, default="assets/ref_images/1.png", help='Path to the source image.')
    parser.add_argument('--driving_video_path', type=str, default="assets/driving_video/6.mp4", help='Path to the driving video.')
    parser.add_argument('--output_path', type=str, default="outputs", help='Path to save the output video.')
    args = parser.parse_args()

    guidance_scale = 3.0
    seed = 43
    num_inference_steps = 10
    sample_size = [480, 720]
    max_frame_num = 10000
    frame_num_per_batch = 49
    overlap_frame_num = 8
    fusion_interval = [3, 8]
    use_interpolation = True
    weight_dtype = torch.bfloat16
    save_path = args.output_path
    generator = torch.Generator(device="cuda").manual_seed(seed)
    model_name = "pretrained_models/SkyReels-A1-5B/"
    siglip_name = "pretrained_models/SkyReels-A1-5B/siglip-so400m-patch14-384"

    lmk_extractor = LMKExtractor()
    processor = FaceAnimationProcessor(checkpoint='pretrained_models/smirk/SMIRK_em1.pt')
    vis = FaceMeshVisualizer2d(forehead_edge=False, draw_head=False, draw_iris=False)
    face_helper = FaceRestoreHelper(upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', device="cuda") 
    
    if use_interpolation:
        frame_inter_model = init_frame_interpolation_model('pretrained_models/film_net/film_net_fp16.pt', device="cuda")

    # siglip visual encoder
    siglip = SiglipVisionModel.from_pretrained(siglip_name)
    siglip_normalize = SiglipImageProcessor.from_pretrained(siglip_name)

    # skyreels a1 model
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_name, 
        subfolder="transformer"
    ).to(weight_dtype)

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_name, 
        subfolder="vae"
    ).to(weight_dtype)

    lmk_encoder = AutoencoderKLCogVideoX.from_pretrained(
        model_name, 
        subfolder="pose_guider",
    ).to(weight_dtype)

    pipe = SkyReelsA1ImagePoseToVideoPipeline.from_pretrained(
        model_name,
        transformer=transformer,
        vae=vae,
        lmk_encoder=lmk_encoder,
        image_encoder=siglip, 
        feature_extractor=siglip_normalize,
        torch_dtype=torch.bfloat16
        )

    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    control_frames = parse_video(args.driving_video_path, max_frame_num, frame_num_per_batch)
    
    # driving video crop face
    driving_video_crop = []
    empty_index = []
    from tqdm import tqdm
    for i, control_frame in enumerate(tqdm(control_frames, desc="Face crop")):
        frame, _, _ = processor.face_crop(control_frame)
        if frame is None:
            print(f'Warning: No face detected in the driving video frame {i}')
            empty_index.append(i)
        else:
            driving_video_crop.append(frame)
    
    control_frames = np.delete(control_frames, empty_index, axis=0)

    video_length = len(driving_video_crop)  # orginal video length

    print(f"orginal video length: {video_length}")

    image = load_image(image=args.image_path)
    image = processor.crop_and_resize(image, sample_size[0], sample_size[1])

    # ref image crop face
    ref_image, x1, y1 = processor.face_crop(np.array(image))
    face_h, face_w, _ = ref_image.shape
    source_image = ref_image
    driving_video = driving_video_crop
    out_frames = processor.preprocess_lmk3d(source_image, driving_video)

    rescale_motions = np.zeros_like(image)[np.newaxis, :].repeat(len(out_frames), axis=0) 
    for ii in range(rescale_motions.shape[0]):
        rescale_motions[ii][y1:y1+face_h, x1:x1+face_w] = out_frames[ii]
    ref_image = cv2.resize(ref_image, (512, 512))
    ref_lmk = lmk_extractor(ref_image[:, :, ::-1])

    ref_img = vis.draw_landmarks_v3((512, 512), (face_w, face_h), ref_lmk['lmks'].astype(np.float32), normed=True)

    first_motion = np.zeros_like(np.array(image))
    first_motion[y1:y1+face_h, x1:x1+face_w] = ref_img
    first_motion = first_motion[np.newaxis, :]

    input_video = rescale_motions[:max_frame_num]

    face_helper.clean_all() 
    face_helper.read_image(np.array(image)[:, :, ::-1])
    face_helper.get_face_landmarks_5(only_center_face=True)
    face_helper.align_warp_face()
    align_face = face_helper.cropped_faces[0]
    image_face = align_face[:, :, ::-1]

    # [F, H, W, C]
    input_video = torch.from_numpy(np.array(input_video)).permute([3, 0, 1, 2]).unsqueeze(0)  # [B, C, F, H, W]
    input_video = input_video / 255
    input_video_all = input_video

    first_motion = torch.from_numpy(np.array(first_motion)).permute([3, 0, 1, 2]).unsqueeze(0)  # [B, C, 1, H, W]
    first_motion = first_motion / 255

    out_samples = []
    padding_frame_num = None
    latents_cache = []

    time_start = time.time()

    for i in range(0, video_length, frame_num_per_batch-1-overlap_frame_num):
        is_first_batch = (i == 0)
        is_last_batch = (i + frame_num_per_batch - 1 >= video_length)

        input_video = input_video_all[:, :, i:i+frame_num_per_batch-1]

        if input_video.shape[2] != frame_num_per_batch-1:
            padding_frame_num = frame_num_per_batch-1 - input_video.shape[2]
            print(f"padding_frame_num: {padding_frame_num}")
            input_video = torch.cat([input_video, torch.repeat_interleave(input_video[:, :, -1].unsqueeze(2), padding_frame_num, dim=2)], dim=2)
        
        input_video = torch.cat([first_motion, input_video], dim=2)

        with torch.no_grad():
            sample, latents_cache = pipe(
                image=image,
                image_face=image_face,
                control_video=input_video,
                prompt="", 
                negative_prompt="",
                height=sample_size[0],
                width=sample_size[1],
                num_frames=frame_num_per_batch,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                is_last_batch=is_last_batch,
                overlap_frame_num=overlap_frame_num,
                fusion_interval=fusion_interval,
                latents_cache=latents_cache
            )
            if use_interpolation:
                if is_first_batch:
                    out_samples = sample.frames[0][1:]
                else:
                    out_samples = smooth_video_transition(out_samples, sample.frames[0][1+overlap_frame_num:], 2, frame_inter_model)
            else:
                out_sample = sample.frames[0][1:] if is_first_batch else sample.frames[0][1+overlap_frame_num:]
                out_samples.extend(out_sample)
            print(f"out_samples len: {len(out_samples)}")

        if is_last_batch:
            break

    if padding_frame_num is not None:
        out_samples = out_samples[:-padding_frame_num]
    
    print(f"Output video length: {len(out_samples)}")
    
    time_end = time.time()
    print(f"time cost: {time_end - time_start} seconds")

    save_path_name = os.path.basename(args.image_path).split(".")[0] + "-" + os.path.basename(args.driving_video_path).split(".")[0]+ ".mp4"

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    video_path = os.path.join(save_path, save_path_name + "-output.mp4")
    export_to_video(out_samples, video_path, fps=12)

    target_h, target_w = sample_size[0], sample_size[1]
    final_images = []
    for q in range(len(out_samples)):
        frame1 = image
        frame2 = crop_and_resize(Image.fromarray(np.array(control_frames[q])).convert("RGB"), target_h, target_w)
        frame3 = Image.fromarray(np.array(out_samples[q])).convert("RGB")

        result = Image.new('RGB', (target_w * 3, target_h))
        result.paste(frame1, (0, 0))
        result.paste(frame2, (target_w, 0))
        result.paste(frame3, (target_w * 2, 0))
        final_images.append(np.array(result))
      
    video_out_path = os.path.join(save_path, save_path_name)
    write_mp4(video_out_path, final_images, fps=12)
        
    add_audio_to_video(video_out_path, args.driving_video_path, video_out_path + f".audio_{frame_num_per_batch}_{overlap_frame_num}.mp4")
    add_audio_to_video(video_path, args.driving_video_path, video_path + f".audio_{frame_num_per_batch}_{overlap_frame_num}.mp4")
