import torch
import os
import numpy as np
from PIL import Image
import glob
import insightface 
import cv2
import subprocess
import argparse
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
from skyreels_a1.skyreels_a1_i2v_pipeline import SkyReelsA1ImagePoseToVideoPipeline
from skyreels_a1.pre_process_lmk3d import FaceAnimationProcessor
from skyreels_a1.src.media_pipe.mp_utils  import LMKExtractor
from skyreels_a1.src.media_pipe.draw_util_2d import FaceMeshVisualizer2d


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

def write_mp4(video_path, samples, fps=14, audio_bitrate="192k"):
    clip = ImageSequenceClip(samples, fps=fps)
    clip.write_videofile(video_path, audio_codec="aac", audio_bitrate=audio_bitrate, 
                         ffmpeg_params=["-crf", "18", "-preset", "slow"])

def init_model(
    model_name: str = "pretrained_models/SkyReels-A1-5B/",
    subfolder: str = "outputs/",
    siglip_path: str = "pretrained_models/SkyReels-A1-5B/siglip-so400m-patch14-384",
    weight_dtype=torch.bfloat16,
    ):

    lmk_extractor = LMKExtractor()
    vis = FaceMeshVisualizer2d(forehead_edge=False, draw_head=False, draw_iris=False,)
    processor = FaceAnimationProcessor(checkpoint='pretrained_models/smirk/SMIRK_em1.pt')

    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        device="cuda",
    ) 

    siglip = SiglipVisionModel.from_pretrained(siglip_path)
    siglip_normalize = SiglipImageProcessor.from_pretrained(siglip_path)
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_name, 
        subfolder="transformer",
    ).to(weight_dtype)

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_name, 
        subfolder="vae"
    ).to(weight_dtype)

    lmk_encoder = AutoencoderKLCogVideoX.from_pretrained(
        model_name, 
        subfolder="pose_guider"
    ).to(weight_dtype)

    pipe = SkyReelsA1ImagePoseToVideoPipeline.from_pretrained(
        model_name,
        transformer = transformer,
        vae = vae,
        lmk_encoder = lmk_encoder,
        image_encoder = siglip, 
        feature_extractor = siglip_normalize,
        torch_dtype=weight_dtype)
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    return pipe, face_helper, processor, lmk_extractor, vis



def generate_video(
    pipe,
    face_helper,
    processor,
    lmk_extractor,
    vis,
    control_video_path: str = None,
    image_path: str = None,
    save_path: str = None,
    guidance_scale=3.0, 
    seed=43,
    num_inference_steps=10,
    sample_size=[480, 720],
    max_frame_num=49,
    weight_dtype=torch.bfloat16,
    ):

    vr = VideoReader(control_video_path)
    fps = vr.get_avg_fps()
    video_length = len(vr)

    duration = video_length / fps 
    target_times = np.arange(0, duration, 1/12)
    frame_indices = (target_times * fps).astype(np.int32)

    frame_indices = frame_indices[frame_indices < video_length]
    control_frames = vr.get_batch(frame_indices).asnumpy()[:(max_frame_num-1)]
    
    out_frames = len(control_frames) - 1
    if  len(control_frames) < max_frame_num:
        video_lenght_add =  max_frame_num - len(control_frames)
    control_frames = np.concatenate(([control_frames[0]]*2, control_frames[1:len(control_frames)-2], [control_frames[-1]] * video_lenght_add), axis=0)

    # driving video crop face
    driving_video_crop = []
    for control_frame in control_frames:
        frame, _, _ = processor.face_crop(control_frame)
        driving_video_crop.append(frame)

    image = load_image(image=image_path)    
    image = crop_and_resize(image, sample_size[0], sample_size[1])

    with torch.no_grad():
        face_helper.clean_all() 
        face_helper.read_image(np.array(image)[:, :, ::-1])
        face_helper.get_face_landmarks_5(only_center_face=True)
        face_helper.align_warp_face()
        if len(face_helper.cropped_faces) == 0:
            return 
        align_face = face_helper.cropped_faces[0]
        image_face = align_face[:, :, ::-1]

    # ref image crop face
    ref_image, x1, y1 = processor.face_crop(np.array(image))
    face_h, face_w, _, = ref_image.shape
    source_image  = ref_image
    driving_video = driving_video_crop
    out_frames = processor.preprocess_lmk3d(source_image, driving_video)

    rescale_motions = np.zeros_like(image)[np.newaxis, :].repeat(48, axis=0)
    for ii in range(rescale_motions.shape[0]):
        rescale_motions[ii][y1:y1+face_h, x1:x1+face_w] = out_frames[ii]
    ref_image = cv2.resize(ref_image, (512, 512))
    ref_lmk = lmk_extractor(ref_image[:, :, ::-1])

    ref_img = vis.draw_landmarks_v3((512, 512), (face_w, face_h), ref_lmk['lmks'].astype(np.float32), normed=True)

    first_motion = np.zeros_like(np.array(image))
    first_motion[y1:y1+face_h, x1:x1+face_w] = ref_img
    first_motion = first_motion[np.newaxis, :]

    motions = np.concatenate([first_motion, rescale_motions])
    input_video = motions[:max_frame_num]

    input_video = input_video[:max_frame_num]
    motions = np.array(input_video)

    # [F, H, W, C]
    input_video = torch.from_numpy(np.array(input_video)).permute([3, 0, 1, 2]).unsqueeze(0)
    input_video = input_video / 255

    out_samples = []

    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        sample = pipe(
            image=image,
            image_face=image_face,
            control_video = input_video,
            height = sample_size[0],
            width = sample_size[1],
            num_frames = 49,
            generator = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,
        )
    out_samples.extend(sample.frames[0][2:])

    # export_to_video(out_samples, save_path, fps=12)
    control_frames = control_frames[1:]
    target_h, target_w = sample_size
    final_images = []
    for i in range(len(out_samples)):
        frame1 = image
        frame2 = crop_and_resize(Image.fromarray(np.array(control_frames[i])).convert("RGB"), target_h, target_w)
        frame3 = Image.fromarray(np.array(out_samples[i])).convert("RGB")
        result = Image.new('RGB', (target_w * 3, target_h))
        result.paste(frame1, (0, 0))
        result.paste(frame2, (target_w, 0))
        result.paste(frame3, (target_w * 2, 0))
        final_images.append(np.array(result))

    write_mp4(save_path, final_images, fps=12)



if __name__ == "__main__":
    control_video_zip = glob.glob("assets/driving_video/*.mp4")
    image_path_zip = glob.glob("assets/ref_images/*.png")

    guidance_scale = 3.0
    seed = 43
    num_inference_steps = 10
    sample_size = [480, 720]
    max_frame_num = 49
    weight_dtype = torch.bfloat16

    save_path = "outputs"

    # init model 
    pipe, face_helper, processor, lmk_extractor, vis = init_model()

    for i in range(len(control_video_zip)): 
        for j in range(len(image_path_zip)):
            generate_video(
                pipe,
                face_helper,
                processor,
                lmk_extractor,
                vis,
                control_video_path=control_video_zip[i],
                image_path=image_path_zip[j],
                save_path=save_path,
                guidance_scale=guidance_scale, 
                seed=seed,
                num_inference_steps=num_inference_steps,
                sample_size=sample_size,
                max_frame_num=max_frame_num,
                weight_dtype=weight_dtype,
            ) 
