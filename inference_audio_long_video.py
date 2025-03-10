import torch
import os
import numpy as np
from PIL import Image
import glob
import insightface 
import cv2
import subprocess
import argparse
import math
from decord import VideoReader
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from insightface.app import FaceAnalysis 
import moviepy.editor as mp
import random


from diffusers.models import AutoencoderKLCogVideoX
from diffusers.utils import export_to_video, load_image
from transformers import AutoModelForDepthEstimation, AutoProcessor, SiglipImageProcessor, SiglipVisionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from skyreels_a1.models.transformer3d import CogVideoXTransformer3DModel
from skyreels_a1.skyreels_a1_i2v_pipeline import SkyReelsA1ImagePoseToVideoPipeline
from skyreels_a1.pre_process_lmk3d import FaceAnimationProcessor
from skyreels_a1.src.media_pipe.mp_utils  import LMKExtractor
from skyreels_a1.src.media_pipe.draw_util_2d import FaceMeshVisualizer2d

from diffposetalk.diffposetalk import DiffPoseTalk


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


def parse_video(driving_frames, max_frame_num, fps=25):

    video_length = len(driving_frames)

    duration = video_length / fps 
    target_times = np.arange(0, duration, 1/12)
    frame_indices = (target_times * fps).astype(np.int32)

    frame_indices = frame_indices[frame_indices < video_length]
    new_driving_frames = []
    for idx in frame_indices:
        new_driving_frames.append(driving_frames[idx])
        if len(new_driving_frames) >= max_frame_num - 1:
            break

    video_lenght_add =  max_frame_num - len(new_driving_frames) - 1
    new_driving_frames = [new_driving_frames[0]]*2 + new_driving_frames[1:len(new_driving_frames)-1] + [new_driving_frames[-1]] * video_lenght_add
    return new_driving_frames


def pad_video(driving_frames, fps=25):
    video_length = len(driving_frames)

    duration = video_length / fps 
    target_times = np.arange(0, duration, 1/12)
    frame_indices = (target_times * fps).astype(np.int32)

    frame_indices = frame_indices[frame_indices < video_length]
    new_driving_frames = []
    for idx in frame_indices:
        new_driving_frames.append(driving_frames[idx])

    pad_length = math.ceil(len(new_driving_frames) / 48) * 48 - len(new_driving_frames)
    new_driving_frames.extend([new_driving_frames[-1]]*pad_length)
    return new_driving_frames, pad_length


def save_video_with_audio(video_path:str, audio_path: str, save_path: str):
    video_clip = mp.VideoFileClip(video_path)
    audio_clip = mp.AudioFileClip(audio_path)

    if audio_clip.duration > video_clip.duration:
        audio_clip = audio_clip.subclip(0, video_clip.duration)

    video_with_audio = video_clip.set_audio(audio_clip)
 
    video_with_audio.write_videofile(save_path, fps=12)

    os.remove(video_path)

    video_clip.close()
    audio_clip.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and image for face animation.")
    parser.add_argument('--image_path', type=str, default="assets/ref_images/1.png", help='Path to the source image.')
    parser.add_argument('--driving_audio_path', type=str, default="assets/driving_audio/1.wav", help='Path to the driving video.')
    parser.add_argument('--output_path', type=str, default="outputs_audio", help='Path to save the output video.')
    args = parser.parse_args()

    guidance_scale = 3.0
    seed = 43
    num_inference_steps = 10
    sample_size = [480, 720]
    max_frame_num = 49
    weight_dtype = torch.bfloat16
    save_path = args.output_path
    generator = torch.Generator(device="cuda").manual_seed(seed)
    model_name = "pretrained_models/SkyReels-A1-5B/"
    siglip_name = "pretrained_models/SkyReels-A1-5B/siglip-so400m-patch14-384"

    lmk_extractor = LMKExtractor()
    processor = FaceAnimationProcessor(checkpoint='pretrained_models/smirk/SMIRK_em1.pt')
    vis = FaceMeshVisualizer2d(forehead_edge=False, draw_head=False, draw_iris=False,)
    face_helper = FaceRestoreHelper(upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', device="cuda",) 

    # diffposetalk init
    diffposetalk = DiffPoseTalk()
     
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
        transformer = transformer,
        vae = vae,
        lmk_encoder = lmk_encoder,
        image_encoder = siglip, 
        feature_extractor = siglip_normalize,
        torch_dtype=torch.bfloat16
        )

    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    image = load_image(image=args.image_path)
    z_image = image = processor.crop_and_resize(image, sample_size[0], sample_size[1])

    # ref image crop face
    ref_image, x1, y1 = processor.face_crop(np.array(image))
    face_h, face_w, _, = ref_image.shape
    source_image  = ref_image

    source_outputs, source_tform, image_original = processor.process_source_image(source_image)
    driving_outputs = diffposetalk.infer_from_file(args.driving_audio_path, source_outputs["shape_params"].view(-1)[:100].detach().cpu().numpy())
    
    out_frames = processor.preprocess_lmk3d_from_coef(source_outputs, source_tform, image_original.shape, driving_outputs)
    out_frames, pad_length = pad_video(out_frames)

    rescale_motions = np.zeros_like(image)[np.newaxis, :].repeat(len(out_frames), axis=0)
    for ii in range(rescale_motions.shape[0]):
        rescale_motions[ii][y1:y1+face_h, x1:x1+face_w] = out_frames[ii]
    ref_image = cv2.resize(ref_image, (512, 512))
    ref_lmk = lmk_extractor(ref_image[:, :, ::-1])

    ref_img = vis.draw_landmarks_v3((512, 512), (face_w, face_h), ref_lmk['lmks'].astype(np.float32), normed=True)

    first_motion = np.zeros_like(np.array(image))
    first_motion[y1:y1+face_h, x1:x1+face_w] = ref_img
    first_motion = first_motion[np.newaxis, :]

    # motions = np.concatenate([first_motion, rescale_motions])
    # input_video = motions[:max_frame_num]

    face_helper.clean_all() 
    face_helper.read_image(np.array(image)[:, :, ::-1])
    face_helper.get_face_landmarks_5(only_center_face=True)
    face_helper.align_warp_face()
    align_face = face_helper.cropped_faces[0]
    image_face = align_face[:, :, ::-1]

    # input_video = input_video[:max_frame_num]
    # motions = np.array(input_video)

    # [F, H, W, C]
    out_samples = []
    for i in range(0, len(rescale_motions), 48):
        # if i == 0:
        #     motions = np.concatenate([first_motion, rescale_motions[i:i+48]])
        # else:
        #     image = out_samples[-1]
        #     motions = rescale_motions[i-1:i+48]
        motions = np.concatenate([first_motion, rescale_motions[i:i+48]])
        input_video = motions
        input_video = torch.from_numpy(np.array(input_video)).permute([3, 0, 1, 2]).unsqueeze(0)
        input_video = input_video / 255


        with torch.no_grad():
            sample = pipe(
                image=image,
                image_face=image_face,
                control_video = input_video,
                prompt = "", 
                negative_prompt = "",
                height = sample_size[0],
                width = sample_size[1],
                num_frames = 49,
                generator = generator,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,
            )
            if i == 0:
                out_samples.extend(sample.frames[0])
            else:
                out_samples.extend(sample.frames[0][1:])
            
    # out_samples = out_samples[2:-pad_length]
    # import pdb; pdb.set_trace()
    if pad_length == 0:
        out_samples = out_samples[1:]
    else:
        out_samples = out_samples[1:-pad_length]
        
    save_path_name = os.path.basename(args.image_path).split(".")[0] + "-" + os.path.basename(args.driving_audio_path).split(".")[0]+ ".mp4"

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    video_path = os.path.join(save_path, save_path_name)

    export_to_video(out_samples, video_path, fps=12)
    save_video_with_audio(video_path, args.driving_audio_path, video_path + ".audio.mp4")
