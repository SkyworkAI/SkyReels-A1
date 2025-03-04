import os
import tempfile

import cv2
import kiui.mesh
import numpy as np

# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # osmesa or egl
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh
# from psbody.mesh import Mesh    


class MeshRenderer:
    def __init__(self, size, fov=16 / 180 * np.pi, camera_pose=None, light_pose=None, black_bg=False):
        # Camera
        self.frustum = {'near': 0.01, 'far': 3.0}
        self.camera = pyrender.PerspectiveCamera(yfov=fov, znear=self.frustum['near'],
                                                 zfar=self.frustum['far'], aspectRatio=1.0)

        # Material
        self.primitive_material = pyrender.material.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[0.3, 0.3, 0.3, 1.0],
            metallicFactor=0.8,
            roughnessFactor=0.8
        )

        # Lighting
        light_color = np.array([1., 1., 1.])
        self.light = pyrender.DirectionalLight(color=light_color, intensity=2)
        self.light_angle = np.pi / 6.0

        # Scene
        self.scene = None
        self._init_scene(black_bg)

        # add camera and lighting
        self._init_camera(camera_pose)
        self._init_lighting(light_pose)

        # Renderer
        self.renderer = pyrender.OffscreenRenderer(*size, point_size=1.0)

    def _init_scene(self, black_bg=False):
        if black_bg:
            self.scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
        else:
            self.scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])

    def _init_camera(self, camera_pose=None):
        if camera_pose is None:
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = np.array([0, 0, 1])
        self.camera_pose = camera_pose.copy()
        self.camera_node = self.scene.add(self.camera, pose=camera_pose)

    def _init_lighting(self, light_pose=None):
        if light_pose is None:
            light_pose = np.eye(4)
            light_pose[:3, 3] = np.array([0, 0, 1])
        self.light_pose = light_pose.copy()

        light_poses = self._get_light_poses(self.light_angle, light_pose)
        self.light_nodes = [self.scene.add(self.light, pose=light_pose) for light_pose in light_poses]

    def set_camera_pose(self, camera_pose):
        self.camera_pose = camera_pose.copy()
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def set_lighting_pose(self, light_pose):
        self.light_pose = light_pose.copy()

        light_poses = self._get_light_poses(self.light_angle, light_pose)
        for light_node, light_pose in zip(self.light_nodes, light_poses):
            self.scene.set_pose(light_node, pose=light_pose)

    def render_mesh(self, v, f, t_center, rot=np.zeros(3), tex_img=None, tex_uv=None,
                    camera_pose=None, light_pose=None):
        # Prepare mesh
        v[:] = cv2.Rodrigues(rot)[0].dot((v - t_center).T).T + t_center
        if tex_img is not None:
            tex = pyrender.Texture(source=tex_img, source_channels='RGB')
            tex_material = pyrender.material.MetallicRoughnessMaterial(baseColorTexture=tex)
            from kiui.mesh import Mesh
            import torch
            mesh = Mesh(
                v=torch.from_numpy(v),
                f=torch.from_numpy(f),
                vt=tex_uv['vt'],
                ft=tex_uv['ft']
            )
            with tempfile.NamedTemporaryFile(suffix='.obj') as f:
                mesh.write_obj(f.name)
                tri_mesh = trimesh.load(f.name, process=False)
            return tri_mesh
            # tri_mesh = self._pyrender_mesh_workaround(mesh)
            render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=tex_material)
        else:
            tri_mesh = trimesh.Trimesh(vertices=v, faces=f)
            render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=self.primitive_material, smooth=True)
        mesh_node = self.scene.add(render_mesh, pose=np.eye(4))

        # Change camera and lighting pose if necessary
        if camera_pose is not None:
            self.set_camera_pose(camera_pose)
        if light_pose is not None:
            self.set_lighting_pose(light_pose)

        # Render
        flags = pyrender.RenderFlags.SKIP_CULL_FACES
        color, depth = self.renderer.render(self.scene, flags=flags)

        # Remove mesh
        self.scene.remove_node(mesh_node)

        return color, depth

    @staticmethod
    def _get_light_poses(light_angle, light_pose):
        light_poses = []
        init_pos = light_pose[:3, 3].copy()

        light_poses.append(light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([light_angle, 0, 0]))[0].dot(init_pos)
        light_poses.append(light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([-light_angle, 0, 0]))[0].dot(init_pos)
        light_poses.append(light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -light_angle, 0]))[0].dot(init_pos)
        light_poses.append(light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([0, light_angle, 0]))[0].dot(init_pos)
        light_poses.append(light_pose.copy())

        return light_poses

    @staticmethod
    def _pyrender_mesh_workaround(mesh):
        # Workaround as pyrender requires number of vertices and uv coordinates to be the same
        with tempfile.NamedTemporaryFile(suffix='.obj') as f:
            mesh.write_obj(f.name)
            tri_mesh = trimesh.load(f.name, process=False)
        return tri_mesh
