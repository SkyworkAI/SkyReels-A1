import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os


import numpy as np
import torch
import torch.nn.functional as F
import os
import cv2

# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]
    
def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(), 
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), 
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    #print('--------')
    #print(camera[0, 1:].abs())
    #print(X[0].abs().mean(0))

    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    #print(X_trans[0].abs().mean(0))
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn

class MP_2_FLAME():
    """
    Convert Mediapipe 52 blendshape scores to FLAME's coefficients 
    """
    def __init__(self, mappings_path):
        self.bs2exp = np.load(os.path.join(mappings_path, 'bs2exp.npy'))
        self.bs2pose = np.load(os.path.join(mappings_path, 'bs2pose.npy'))
        self.bs2eye = np.load(os.path.join(mappings_path, 'bs2eye.npy'))

    def convert(self, blendshape_scores : np.array):
        # blendshape_scores: [N, 52]

        # Calculate expression, pose, and eye_pose using the mappings
        exp = blendshape_scores @ self.bs2exp 
        pose = blendshape_scores @ self.bs2pose 
        pose[0, :3] = 0  # we do not support head rotation yet
        eye_pose = blendshape_scores @ self.bs2eye

        return exp, pose, eye_pose

class MediaPipeUtils:
    def __init__(self, model_asset_path='pretrained_models/mediapipe/face_landmarker.task', mappings_path='pretrained_models/mediapipe/'):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1,
                                               min_face_detection_confidence=0.1,
                                               min_face_presence_confidence=0.1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.mp2flame = MP_2_FLAME(mappings_path=mappings_path)

    def run_mediapipe(self, image):
        image_numpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)
        detection_result = self.detector.detect(image)

        if len(detection_result.face_landmarks) == 0:
            print('No face detected')
            return None

        blend_scores = detection_result.face_blendshapes[0]
        blend_scores = np.array(list(map(lambda l: l.score, blend_scores)), dtype=np.float32).reshape(1, 52)
        exp, pose, eye_pose = self.mp2flame.convert(blendshape_scores=blend_scores)

        face_landmarks = detection_result.face_landmarks[0]
        face_landmarks_numpy = np.zeros((478, 3))

        for i, landmark in enumerate(face_landmarks):
            face_landmarks_numpy[i] = [landmark.x * image.width, landmark.y * image.height, landmark.z]

        return face_landmarks_numpy, exp, pose, eye_pose
