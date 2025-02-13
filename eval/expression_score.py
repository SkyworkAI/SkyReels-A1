import os 
import os 
import torch 
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image
from torchvision import models, transforms
from curricularface import get_model
import cv2
import numpy as np 
import numpy

def pad_np_bgr_image(np_image, scale=1.25):
    assert scale >= 1.0, "scale should be >= 1.0"
    pad_scale = scale - 1.0
    h, w = np_image.shape[:2]
    top = bottom = int(h * pad_scale)
    left = right = int(w * pad_scale)
    return cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128)), (left, top)


def sample_video_frames(video_path,):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    frame_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # if frame.shape[1] > 1024:
                # frame = frame[:, 1440:, :]  
                # print(frame.shape) 
            frame = cv2.resize(frame, (720, 480))
            # print(frame.shape)
            frames.append(frame)
    cap.release()
    return frames


def get_face_keypoints(face_model, image_bgr):
    face_info = face_model.get(image_bgr)
    if len(face_info) > 0:
        return sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
    return None

def process_image(face_model, image_path):
    if isinstance(image_path, str):
        np_faceid_image = np.array(Image.open(image_path).convert("RGB"))
    elif isinstance(image_path, numpy.ndarray):
        np_faceid_image = image_path
    else:
        raise TypeError("image_path should be a string or PIL.Image.Image object")

    image_bgr = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)

    face_info = get_face_keypoints(face_model, image_bgr)
    if face_info is None:
        padded_image, sub_coord = pad_np_bgr_image(image_bgr)
        face_info = get_face_keypoints(face_model, padded_image)
        if face_info is None:
            print("Warning: No face detected in the image. Continuing processing...")
            return None
        face_kps = face_info['kps']
        face_kps -= np.array(sub_coord)
    else:
        face_kps = face_info['kps']
    return face_kps 

def process_video(video_path, face_arc_model):
    video_frames = sample_video_frames(video_path,)
    print(len(video_frames))
    kps_list = []
    for frame in video_frames:
        # Convert to RGB once at the beginning
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        kps = process_image(face_arc_model, frame_rgb)
        if kps is None:
            return None 
        # print(kps)
        kps_list.append(kps) 
    return kps_list 


def calculate_l1_distance(list1, list2):
    """
    计算两个列表的 L1 距离
    :param list1: 第一个列表，形状为 (5, 2)
    :param list2: 第二个列表，形状为 (5, 2)
    :return: L1 距离
    """
    # 将列表转换为 NumPy 数组
    list1 = np.array(list1)
    list2 = np.array(list2)

    # 计算每对点的 L1 距离
    l1_distances = np.abs(list1 - list2).sum(axis=1)

    # 返回所有点的 L1 距离之和
    return l1_distances.sum()


def calculate_kps(list1, list2): 
    distance_list = []
    for kps1 in list1:
        min_dis = (480 + 720) * 5 + 1 
        for kps2 in list2: 
            min_dis = min(min_dis, calculate_l1_distance(kps1, kps2)) 
        distance_list.append(min_dis/(480+720)/10)
    return sum(distance_list)/len(distance_list)


def main():
    device = "cuda" 
    # data_path = "data/SkyActor" 
    # data_path = "data/LivePotraits"
    # data_path = "data/Actor-One"
    data_path = "data/FollowYourEmoji"
    img_path = "/maindata/data/shared/public/rui.wang/act_review/driving_video"
    pre_tag = False 
    mp4_list = os.listdir(data_path) 
    print(mp4_list) 
    
    img_list = []
    video_list = []
    for mp4 in mp4_list:
        if "mp4" not in mp4:
            continue 
        if pre_tag: 
            png_path = mp4.split('.')[0].split('--')[1] + ".mp4" 
        else:
            if "-" in mp4:
                png_path = mp4.split('.')[0].split('-')[0] + ".mp4" 
            else: 
                png_path = mp4.split('.')[0].split('_')[0] + ".mp4" 
        img_list.append(os.path.join(img_path, png_path))
        video_list.append(os.path.join(data_path, mp4))        
    print(img_list)
    print(video_list[0]) 

    model_path = "/maindata/data/shared/public/zhengcong.fei/ckpts/cogvideox1.5/consistent_id" 
    face_arc_path = os.path.join(model_path, "face_encoder")
    face_cur_path = os.path.join(face_arc_path, "glint360k_curricular_face_r101_backbone.bin") 

    # Initialize FaceEncoder model for face detection and embedding extraction
    face_arc_model = FaceAnalysis(root=face_arc_path, providers=['CUDAExecutionProvider'])
    face_arc_model.prepare(ctx_id=0, det_size=(320, 320))

    expression_list = []
    for i in range(len(img_list)):  
        print("number: ", str(i), " total: ", len(img_list), data_path) 
        kps_1 = process_video(video_list[i], face_arc_model)
        kps_2 = process_video(img_list[i], face_arc_model) 
        if kps_1 is None or kps_2 is None:
            continue 

        dis = calculate_kps(kps_1, kps_2)
        print(dis)
        expression_list.append(dis)
        # break 
    
    print("kps", sum(expression_list)/ len(expression_list))



main()