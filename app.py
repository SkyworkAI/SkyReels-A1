import gradio as gr
from scripts.demo import init_model, generate_video, crop_and_resize
import os
import os.path as osp
import stat
from datetime import datetime
import torch
import numpy as np
from diffusers.utils import export_to_video, load_image

os.environ['GRADIO_TEMP_DIR'] = 'tmp'

example_portrait_dir = "assets/ref_images"
example_video_dir = "assets/driving_video"


pipe, face_helper, processor, lmk_extractor, vis = init_model()
# Gradio interface using Interface
with gr.Blocks() as demo:
    gr.Markdown("""
        <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               SkyReels-A1: Expressive Portrait Animation in Video Diffusion Transformers
        </div>
        <div style="text-align: center;">
               <a href="">🤗 SkyReels-A1-5B Model Hub</a> |
               <a href="https://github.com/SkyworkAI/SkyReels-A1">🌐 Github</a> |
               <a href="">📜 arxiv </a>
        </div>
    """)

    with gr.Row():  # 创建一个水平排列的行
        with gr.Accordion(open=True, label="Portrait Image"):
            image_input = gr.Image(type="filepath")
            gr.Examples(
                examples=[
                    [osp.join(example_portrait_dir, "1.png")],
                    [osp.join(example_portrait_dir, "2.png")],
                    [osp.join(example_portrait_dir, "3.png")],
                    [osp.join(example_portrait_dir, "4.png")],
                    [osp.join(example_portrait_dir, "5.png")],
                    [osp.join(example_portrait_dir, "6.png")],
                    [osp.join(example_portrait_dir, "7.png")],
                    [osp.join(example_portrait_dir, "8.png")],
                ],
                inputs=[image_input],
                cache_examples=False,
            )
        with gr.Accordion(open=True, label="Driving Video"):
            control_video_input = gr.Video()
            gr.Examples(
                examples=[
                    [osp.join(example_video_dir, "1.mp4")],
                    [osp.join(example_video_dir, "2.mp4")],
                    [osp.join(example_video_dir, "3.mp4")],
                    [osp.join(example_video_dir, "4.mp4")],
                    [osp.join(example_video_dir, "5.mp4")],
                    [osp.join(example_video_dir, "6.mp4")],
                    [osp.join(example_video_dir, "7.mp4")],
                    [osp.join(example_video_dir, "8.mp4")],
                ],
                inputs=[control_video_input],
                cache_examples=False,
                )

    def face_check(image_path):
        image = load_image(image=image_path)    
        image = crop_and_resize(image, 480, 720)

        with torch.no_grad():
            face_helper.clean_all() 
            face_helper.read_image(np.array(image)[:, :, ::-1])
            face_helper.get_face_landmarks_5(only_center_face=True)
            face_helper.align_warp_face()
            if len(face_helper.cropped_faces) == 0:
                return False
            face = face_helper.det_faces
            face_w = int(face[2] - face[0])
            if face_w < 50:
                return False
        return True


    def gradio_generate_video(control_video_path, image_path, progress=gr.Progress(track_tqdm=True)):
        try:
            save_dir = "./outputs/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"generated_video_{current_time}.mp4")
            print(control_video_path, image_path)

            face = face_check(image_path)
            if face == False:
                return "Face too small or no face.", None, None

            generate_video(
                pipe,
                face_helper,
                processor, 
                lmk_extractor, 
                vis,
                control_video_path=control_video_path,
                image_path=image_path,
                save_path=save_path,
                guidance_scale=3, 
                seed=43,
                num_inference_steps=20,
                sample_size=[480, 720],
                max_frame_num=49,
            )

            print("finished.")
            print(save_path)
            if not os.path.exists(save_path):
                print("Error: Video file not found")
                return "Error: Video file not found", None

            video_update = gr.update(visible=True, value=save_path)
            return "Video generated successfully.", save_path, video_update
        except Exception as e:
            return f"Error occurred: {str(e)}", None, None

    
    generate_button = gr.Button("Generate Video")
    output_text = gr.Textbox(label="Output")
    output_video = gr.Video(label="Output Video") 
    with gr.Row():
        download_video_button = gr.File(label="📥 Download Video", visible=False)

    generate_button.click(
            gradio_generate_video,
            inputs=[
                control_video_input,
                image_input
            ],
            outputs=[output_text, output_video, download_video_button],  # 更新输出以包含视频
            show_progress=True,
        )


if __name__ == "__main__":
    # demo.queue(concurrency_count=8)
    demo.launch(share=True, enable_queue=True)
