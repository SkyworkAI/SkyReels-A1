import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
import pickle
import chumpy as ch
import cv2
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skyreels_a1.src.utils.mediapipe_utils import face_vertices, vertex_normals, batch_orth_proj
from skyreels_a1.src.media_pipe.draw_util import FaceMeshVisualizer
from mediapipe.framework.formats import landmark_pb2

def keep_vertices_and_update_faces(faces, vertices_to_keep):
    """
    Keep specified vertices in the mesh and update the faces.
    """
    if isinstance(vertices_to_keep, list) or isinstance(vertices_to_keep, np.ndarray):
        vertices_to_keep = torch.tensor(vertices_to_keep, dtype=torch.long)

    vertices_to_keep = torch.unique(vertices_to_keep)
    max_vertex_index = faces.max().long().item() + 1

    mask = torch.zeros(max_vertex_index, dtype=torch.bool)
    mask[vertices_to_keep] = True

    new_vertex_indices = torch.full((max_vertex_index,), -1, dtype=torch.long)
    new_vertex_indices[mask] = torch.arange(len(vertices_to_keep))

    valid_faces_mask = (new_vertex_indices[faces] != -1).all(dim=1)
    filtered_faces = faces[valid_faces_mask]
    updated_faces = new_vertex_indices[filtered_faces]

    return updated_faces

def predict_landmark_position(ref_points, relative_coords):
    """
    Predict the new position of the eyeball based on reference points and relative coordinates.
    """
    left_corner = ref_points[0]
    right_corner = ref_points[8]
    
    eye_center = (left_corner + right_corner) / 2
    eye_width_vector = right_corner - left_corner
    eye_width = np.linalg.norm(eye_width_vector)
    eye_direction = eye_width_vector / eye_width
    eye_vertical = np.array([-eye_direction[1], eye_direction[0]])
    
    predicted_pos = eye_center + \
                   (eye_width/2) * relative_coords[0] * eye_direction + \
                   (eye_width/2) * relative_coords[1] * eye_vertical
    
    return predicted_pos

def mesh_points_by_barycentric_coordinates(mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords):
    """
    Evaluation 3d points given mesh and landmark embedding
    """
    dif1 = ch.vstack([
        (mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
        (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
        (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)
    ]).T
    return dif1

class Renderer(nn.Module):
    def __init__(self, render_full_head=False, obj_filename='pretrained_models/FLAME/head_template.obj'):
        super(Renderer, self).__init__()
        self.image_size = 224
        self.mediapipe_landmark_embedding = np.load("pretrained_models/smirk/mediapipe_landmark_embedding.npz")
        self.vis = FaceMeshVisualizer(forehead_edge=False)
        
        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]   # (N, F, 3)
        faces = faces.verts_idx[None,...]

        self.render_full_head = render_full_head

        red_color = torch.tensor([255, 0, 0])[None, None, :].float() / 255.
        transparent_color = torch.tensor([0, 0, 0])[None, None, :].float()
        colors = transparent_color.repeat(1, 5023, 1)

        flame_masks = pickle.load(
            open('pretrained_models/FLAME/FLAME_masks.pkl', 'rb'),
            encoding='latin1')
        self.flame_masks = flame_masks

        self.register_buffer('faces', faces)

        face_colors = face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)
        
        self.register_buffer('raw_uvcoords', uvcoords)

        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        pi = np.pi
        constant_factor = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                           ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                           (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)

    def forward(self, vertices, cam_params, source_tform=None, tform_512=None, weights_468=None, weights_473=None,shape = None, **landmarks):
        transformed_vertices = batch_orth_proj(vertices, cam_params)
        transformed_vertices[:, :, 1:] = -transformed_vertices[:, :, 1:]

        transformed_landmarks = {}
        for key in landmarks.keys():
            transformed_landmarks[key] = batch_orth_proj(landmarks[key], cam_params)
            transformed_landmarks[key][:, :, 1:] = - transformed_landmarks[key][:, :, 1:]
            transformed_landmarks[key] = transformed_landmarks[key][...,:2]

        rendered_img = self.render(vertices, transformed_vertices, source_tform, tform_512, weights_468, weights_473,shape)
        outputs = {
            'rendered_img': rendered_img,
            'transformed_vertices': transformed_vertices
        }
        outputs.update(transformed_landmarks)

        return outputs
    
    def _calculate_eye_landmarks(self, landmark_list_pixlevel, weights_468, weights_473, source_tform):
        #  [np.array([x_relative, y_relative]),target_point,ref_points] 根据当前的new_landmarks，根据target_point， 利用映射变换，计算出眼部landmarks
       
       
       import pdb; pdb.set_trace()
       pass


    def render(self, vertices, transformed_vertices, source_tform, tform_512, weights_468, weights_473, shape):
        # batch_size = vertices.shape[0]
        transformed_vertices[:,:,2] += 10  # Z-axis offset
        
        # colors = self.face_colors.expand(batch_size, -1, -1, -1)
        # rendering = self.rasterize(transformed_vertices, self.faces.expand(batch_size, -1, -1), colors)
        
        v_selected = self._calculate_landmark_points(transformed_vertices)
        v_selected_tensor = torch.tensor(v_selected, dtype=torch.float32, device=transformed_vertices.device) #torch.Size([113, 3])
        # import pdb; pdb.set_trace()
        new_landmarks, landmark_list_pixlevel = self._create_landmark_list(v_selected_tensor, source_tform, shape)
        # 基于weights_468和weights_473，计算眼部landmarks
        left_eye_point_indices = weights_468[3]
        right_eye_point_indices = weights_473[3]
        # 遍历每个索引以找到其在 index_mapping 中的位置
        left_eye_point_indices = [self.vis.index_mapping.index(idx) for idx in left_eye_point_indices]
        right_eye_point_indices = [self.vis.index_mapping.index(idx) for idx in right_eye_point_indices]

        left_eye_point = [landmark_list_pixlevel[idx] for idx in left_eye_point_indices]
        right_eye_point = [landmark_list_pixlevel[idx] for idx in right_eye_point_indices]
        # import pdb; pdb.set_trace()
        # weights_468[2].shape = (16, 2)    
        M_affine_left, _ = cv2.estimateAffine2D(np.array(weights_468[2], dtype=np.float32), np.array(left_eye_point, dtype=np.float32))
        M_affine_right, _ = cv2.estimateAffine2D(np.array(weights_473[2], dtype=np.float32), np.array(right_eye_point, dtype=np.float32))

        # 计算瞳孔点
        pupil_left_eye = cv2.transform(weights_468[1].reshape(1, 1, 2), M_affine_left).reshape(-1)
        pupil_right_eye = cv2.transform(weights_473[1].reshape(1, 1, 2), M_affine_right).reshape(-1)

        # left_eye_point, right_eye_point = self._calculate_eye_landmarks(landmark_list_pixlevel, weights_468, weights_473, source_tform)
        # left_eye_point, right_eye_point = self._process_eye_landmarks(transformed_vertices, source_tform)
        # import pdb; pdb.set_trace()
        return self._generate_final_image(new_landmarks, pupil_left_eye, pupil_right_eye, shape)
        # return self._generate_final_image(new_landmarks, left_eye_point, right_eye_point, shape)

    def _calculate_landmark_points(self, transformed_vertices):
        lmk_b_coords = self.mediapipe_landmark_embedding['lmk_b_coords']
        lmk_face_idx = self.mediapipe_landmark_embedding['lmk_face_idx']
        
        base_points = mesh_points_by_barycentric_coordinates(
            transformed_vertices.detach().cpu().numpy()[0],
            self.faces.detach().cpu().numpy()[0],
            lmk_face_idx, lmk_b_coords
        )
        
        RIGHT_EYE_INDICES = [4543, 4511, 4479, 4575]
        LEFT_EYE_INDICES = [3997, 3965, 3933, 4020]
        return np.concatenate([
            base_points,
            transformed_vertices.detach().cpu().numpy()[0][RIGHT_EYE_INDICES],
            transformed_vertices.detach().cpu().numpy()[0][LEFT_EYE_INDICES]
        ], axis=0)

    def _create_landmark_list(self, vertices, transform, shape):
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        landmark_list_pixlevel = []
        for v in vertices:
            img_x = (v[0] + 1) * 0.5 * self.image_size
            img_y = (v[1] + 1) * 0.5 * self.image_size
            projected = np.dot(transform.inverse.params, [img_x.cpu().numpy(), img_y.cpu().numpy(), 1.0])
            landmark_list_pixlevel.append((projected[0], projected[1]))
            landmark = landmark_list.landmark.add()
            landmark.x = projected[0] / shape[1]
            landmark.y = projected[1] / shape[0]
            landmark.z = 1.0
        return landmark_list, landmark_list_pixlevel

    def _process_eye_landmarks(self, vertices, transform):
        def project_eye_point(vertex_idx):
            x = (vertices[0, vertex_idx, 0] + 1) * 0.5 * self.image_size
            y = (vertices[0, vertex_idx, 1] + 1) * 0.5 * self.image_size
            # import pdb; pdb.set_trace()
            projected = np.dot(transform.inverse.params, [x.cpu().numpy(), y.cpu().numpy(), 1.0])
            return (int(projected[0]), int(projected[1]))

        return (
            project_eye_point(4051),  # Left eye index
            project_eye_point(4597)   # Right eye index
        )

    def _generate_final_image(self, landmarks, left_eye, right_eye, shape):
        image = np.zeros([shape[0], shape[1], 3], dtype=np.uint8)
        
        self.vis.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=self.vis.face_connection_spec.keys(),
            landmark_drawing_spec=None,
            connection_drawing_spec=self.vis.face_connection_spec
        )
        
        self._draw_eye_markers(image, np.array(left_eye, dtype=np.int32), np.array(right_eye, dtype=np.int32))
        self._draw_landmark_58(image, landmarks, shape)
        return np.copy(image)

    def _draw_eye_markers(self, image, left_eye, right_eye):
        y, x = left_eye[1]-1, left_eye[0]-1
        image[y:y+3, x:x+3] = [10, 200, 250]
        
        y, x = right_eye[1]-1, right_eye[0]-1
        image[y:y+3, x:x+3] = [250, 200, 10]

    def _draw_landmark_58(self, image, landmarks, shape):
        if len(landmarks.landmark) > 57:
            point = landmarks.landmark[57]
            x = int(point.x * shape[1])
            y = int(point.y * shape[0])
            image[y-2:y+3, x-2:x+3] = [255, 255, 255]

    def rasterize(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]

        if h is None and w is None:
            image_size = self.image_size
        else:
            image_size = [h, w]
            if h>w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=False,
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone(); attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
        return pixel_vals

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
                N[:,0]*0.+1., N[:,0], N[:,1], \
                N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
                N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
                ], 
                1) # [bz, 9, h, w]
        sh = sh*self.constant_factor[None,:,None,None]
        shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_positions[:,:,None,:] - vertices[:,None,:,:], dim=3)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)
    

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)



    def render_multiface(self, vertices, transformed_vertices, faces):
        
        batch_size = vertices.shape[0]

        light_positions = torch.tensor(
            [
            [-1,-1,-1],
            [1,-1,-1],
            [-1,+1,-1],
            [1,+1,-1],
            [0,0,-1]
            ]
        )[None,:,:].expand(batch_size, -1, -1).float()

        light_intensities = torch.ones_like(light_positions).float()*1.7
        lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        normals = vertex_normals(vertices, faces) 
        face_normals = face_vertices(normals, faces)
        
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, transformed_vertices.shape[1]+1, 1).float()/255.
        colors = colors.cuda()
        face_colors = face_vertices(colors, faces[0].unsqueeze(0))
        
        colors = face_colors.expand(batch_size, -1, -1, -1)

        attributes = torch.cat([colors,
                                face_normals], 
                                -1)
        rendering = self.rasterize(transformed_vertices, faces, attributes)
        
        albedo_images = rendering[:, :3, :, :]

        normal_images = rendering[:, 3:6, :, :]

        shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()        
        shaded_images = albedo_images*shading_images
        
        return shaded_images
