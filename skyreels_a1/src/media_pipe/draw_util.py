import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

class FaceMeshVisualizer:
    def __init__(self, forehead_edge=False, iris_edge=False, iris_point=False):
        self.mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_mesh = mp_face_mesh
        self.forehead_edge = forehead_edge

        DrawingSpec = mp.solutions.drawing_styles.DrawingSpec
        f_thick = 1
        f_rad = 1
        right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
        right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
        right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
        left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
        left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
        left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
        # head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)
        head_draw = DrawingSpec(color=(0, 0, 0), thickness=f_thick, circle_radius=f_rad)
        
        mouth_draw_obl = DrawingSpec(color=(10, 180, 20), thickness=f_thick, circle_radius=f_rad)
        mouth_draw_obr = DrawingSpec(color=(20, 10, 180), thickness=f_thick, circle_radius=f_rad)
        
        mouth_draw_ibl = DrawingSpec(color=(100, 100, 30), thickness=f_thick, circle_radius=f_rad)
        mouth_draw_ibr = DrawingSpec(color=(100, 150, 50), thickness=f_thick, circle_radius=f_rad)
        
        mouth_draw_otl = DrawingSpec(color=(20, 80, 100), thickness=f_thick, circle_radius=f_rad)
        mouth_draw_otr = DrawingSpec(color=(80, 100, 20), thickness=f_thick, circle_radius=f_rad)
        
        mouth_draw_itl = DrawingSpec(color=(120, 100, 200), thickness=f_thick, circle_radius=f_rad)
        mouth_draw_itr = DrawingSpec(color=(150 ,120, 100), thickness=f_thick, circle_radius=f_rad)

        self.pupil_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}
        
        # FACEMESH_LIPS_OUTER_BOTTOM_LEFT = [(61,146),(146,91),(91,181),(181,84),(84,17)]
        # FACEMESH_LIPS_OUTER_BOTTOM_RIGHT = [(17,314),(314,405),(405,321),(321,375),(375,291)]
        
        # FACEMESH_LIPS_INNER_BOTTOM_LEFT = [(78,95),(95,88),(88,178),(178,87),(87,14)]
        # FACEMESH_LIPS_INNER_BOTTOM_RIGHT = [(14,317),(317,402),(402,318),(318,324),(324,308)]
        
        # FACEMESH_LIPS_OUTER_TOP_LEFT = [(61,185),(185,40),(40,39),(39,37),(37,0)]
        # FACEMESH_LIPS_OUTER_TOP_RIGHT = [(0,267),(267,269),(269,270),(270,409),(409,291)]
        
        # FACEMESH_LIPS_INNER_TOP_LEFT = [(78,191),(191,80),(80,81),(81,82),(82,13)]
        # FACEMESH_LIPS_INNER_TOP_RIGHT = [(13,312),(312,311),(311,310),(310,415),(415,308)]
        
        # FACEMESH_CUSTOM_FACE_OVAL = [(176, 149), (150, 136), (356, 454), (58, 132), (152, 148), (361, 288), (251, 389), (132, 93), (389, 356), (400, 377), (136, 172), (377, 152), (323, 361), (172, 58), (454, 323), (365, 379), (379, 378), (148, 176), (93, 234), (397, 365), (149, 150), (288, 397), (234, 127), (378, 400), (127, 162), (162, 21)]

        index_mapping = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 46, 52, 53,
                        55, 63, 65, 66, 70, 105, 107, 249, 263, 362, 373, 374, 380,
                        381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 7, 33, 133,
                        144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
                        168, 6, 197, 195, 5, 4, 129, 98, 97, 2, 326, 327, 358,
                        0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84,
                        87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
                        308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
                        415]#, 469, 470, 471, 472, 474, 475, 476, 477]

        self.index_mapping = index_mapping

        def safe_index(mapping, value):
            try:
                return mapping.index(value)
            except ValueError:
                return None

        # 使用新的landmark索引映射
        FACEMESH_LIPS_OUTER_BOTTOM_LEFT = [
            (a, b) for a, b in [
                (safe_index(index_mapping, 61), safe_index(index_mapping, 146)),
                (safe_index(index_mapping, 146), safe_index(index_mapping, 91)),
                (safe_index(index_mapping, 91), safe_index(index_mapping, 181)),
                (safe_index(index_mapping, 181), safe_index(index_mapping, 84)),
                (safe_index(index_mapping, 84), safe_index(index_mapping, 17))
            ] if a is not None and b is not None
        ]

        FACEMESH_LIPS_OUTER_BOTTOM_RIGHT = [
            (a, b) for a, b in [
                (safe_index(index_mapping, 17), safe_index(index_mapping, 314)),
                (safe_index(index_mapping, 314), safe_index(index_mapping, 405)),
                (safe_index(index_mapping, 405), safe_index(index_mapping, 321)),
                (safe_index(index_mapping, 321), safe_index(index_mapping, 375)),
                (safe_index(index_mapping, 375), safe_index(index_mapping, 291))
            ] if a is not None and b is not None
        ]

        FACEMESH_LIPS_INNER_BOTTOM_LEFT = [
            (a, b) for a, b in [
                (safe_index(index_mapping, 78), safe_index(index_mapping, 95)),
                (safe_index(index_mapping, 95), safe_index(index_mapping, 88)),
                (safe_index(index_mapping, 88), safe_index(index_mapping, 178)),
                (safe_index(index_mapping, 178), safe_index(index_mapping, 87)),
                (safe_index(index_mapping, 87), safe_index(index_mapping, 14))
            ] if a is not None and b is not None
        ]

        FACEMESH_LIPS_INNER_BOTTOM_RIGHT = [
            (a, b) for a, b in [
                (safe_index(index_mapping, 14), safe_index(index_mapping, 317)),
                (safe_index(index_mapping, 317), safe_index(index_mapping, 402)),
                (safe_index(index_mapping, 402), safe_index(index_mapping, 318)),
                (safe_index(index_mapping, 318), safe_index(index_mapping, 324)),
                (safe_index(index_mapping, 324), safe_index(index_mapping, 308))
            ] if a is not None and b is not None
        ]

        FACEMESH_LIPS_OUTER_TOP_LEFT = [
            (a, b) for a, b in [
                (safe_index(index_mapping, 61), safe_index(index_mapping, 185)),
                (safe_index(index_mapping, 185), safe_index(index_mapping, 40)),
                (safe_index(index_mapping, 40), safe_index(index_mapping, 39)),
                (safe_index(index_mapping, 39), safe_index(index_mapping, 37)),
                (safe_index(index_mapping, 37), safe_index(index_mapping, 0))
            ] if a is not None and b is not None
        ]

        FACEMESH_LIPS_OUTER_TOP_RIGHT = [
            (a, b) for a, b in [
                (safe_index(index_mapping, 0), safe_index(index_mapping, 267)),
                (safe_index(index_mapping, 267), safe_index(index_mapping, 269)),
                (safe_index(index_mapping, 269), safe_index(index_mapping, 270)),
                (safe_index(index_mapping, 270), safe_index(index_mapping, 409)),
                (safe_index(index_mapping, 409), safe_index(index_mapping, 291))
            ] if a is not None and b is not None
        ]

        FACEMESH_LIPS_INNER_TOP_LEFT = [
            (a, b) for a, b in [
                (safe_index(index_mapping, 78), safe_index(index_mapping, 191)),
                (safe_index(index_mapping, 191), safe_index(index_mapping, 80)),
                (safe_index(index_mapping, 80), safe_index(index_mapping, 81)),
                (safe_index(index_mapping, 81), safe_index(index_mapping, 82)),
                (safe_index(index_mapping, 82), safe_index(index_mapping, 13))
            ] if a is not None and b is not None
        ]

        FACEMESH_LIPS_INNER_TOP_RIGHT = [
            (a, b) for a, b in [
                (safe_index(index_mapping, 13), safe_index(index_mapping, 312)),
                (safe_index(index_mapping, 312), safe_index(index_mapping, 311)),
                (safe_index(index_mapping, 311), safe_index(index_mapping, 310)),
                (safe_index(index_mapping, 310), safe_index(index_mapping, 415)),
                (safe_index(index_mapping, 415), safe_index(index_mapping, 308))
            ] if a is not None and b is not None
        ]


        FACEMESH_EYE_RIGHT = [
            (a, b) for a, b in [
                (safe_index(index_mapping, 144), safe_index(index_mapping, 145)),
                (safe_index(index_mapping, 145), safe_index(index_mapping, 153)),
                (safe_index(index_mapping, 153), safe_index(index_mapping, 154)),
                (safe_index(index_mapping, 154), safe_index(index_mapping, 155)),
                (safe_index(index_mapping, 155), safe_index(index_mapping, 157)),
                (safe_index(index_mapping, 157), safe_index(index_mapping, 158)),
                (safe_index(index_mapping, 158), safe_index(index_mapping, 159)),
                (safe_index(index_mapping, 159), safe_index(index_mapping, 160)),
                (safe_index(index_mapping, 160), safe_index(index_mapping, 161)),
                (safe_index(index_mapping, 161), safe_index(index_mapping, 33)),
                (safe_index(index_mapping, 33), safe_index(index_mapping, 7)),
                (safe_index(index_mapping, 7), safe_index(index_mapping, 163)),
                
                (safe_index(index_mapping, 46), safe_index(index_mapping, 53)),
                (safe_index(index_mapping, 53), safe_index(index_mapping, 52)),
                (safe_index(index_mapping, 52), safe_index(index_mapping, 65)),
                (safe_index(index_mapping, 65), safe_index(index_mapping, 55)),

                (safe_index(index_mapping, 107), safe_index(index_mapping, 66)),
                (safe_index(index_mapping, 66), safe_index(index_mapping, 105)),
                (safe_index(index_mapping, 105), safe_index(index_mapping, 63)),
                (safe_index(index_mapping, 63), safe_index(index_mapping, 70)),

            ] if a is not None and b is not None
        ]
        # index_mapping = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 46, 52, 53,
        #                 55, 63, 65, 66, 70, 105, 107, 249, 263, 362, 373, 374, 380,
        #                 381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 7, 33, 133,
        #                 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
        #                 168, 6, 197, 195, 5, 4, 129, 98, 97, 2, 326, 327, 358,
        #                 0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84,
        #                 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
        #                 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
        #                 415]

        FACEMESH_EYE_LEFT = [
            (a, b) for a, b in [
                (safe_index(index_mapping, 362), safe_index(index_mapping, 382)),
                (safe_index(index_mapping, 382), safe_index(index_mapping, 381)),
                (safe_index(index_mapping, 381), safe_index(index_mapping, 380)),
                (safe_index(index_mapping, 380), safe_index(index_mapping, 374)),
                (safe_index(index_mapping, 374), safe_index(index_mapping, 373)),
                (safe_index(index_mapping, 373), safe_index(index_mapping, 390)),
                (safe_index(index_mapping, 390), safe_index(index_mapping, 249)),
                (safe_index(index_mapping, 249), safe_index(index_mapping, 263)),
                (safe_index(index_mapping, 263), safe_index(index_mapping, 466)),
                (safe_index(index_mapping, 466), safe_index(index_mapping, 388)),
                (safe_index(index_mapping, 388), safe_index(index_mapping, 387)),
                (safe_index(index_mapping, 387), safe_index(index_mapping, 386)),
                (safe_index(index_mapping, 386), safe_index(index_mapping, 385)),
                (safe_index(index_mapping, 385), safe_index(index_mapping, 384)),
                (safe_index(index_mapping, 384), safe_index(index_mapping, 398)),
                (safe_index(index_mapping, 398), safe_index(index_mapping, 362)),
                
                # (safe_index(index_mapping, 285), safe_index(index_mapping, 295)),
                # (safe_index(index_mapping, 295), safe_index(index_mapping, 282)),
                # (safe_index(index_mapping, 282), safe_index(index_mapping, 283)),
                # (safe_index(index_mapping, 283), safe_index(index_mapping, 276)),

                # (safe_index(index_mapping, 336), safe_index(index_mapping, 296)),
                # (safe_index(index_mapping, 296), safe_index(index_mapping, 334)),
                # (safe_index(index_mapping, 334), safe_index(index_mapping, 293)),
                # (safe_index(index_mapping, 293), safe_index(index_mapping, 300)),

            ] if a is not None and b is not None
        ]


        FACEMESH_EYE_LEFT_new = [(0,267),(267,269),(269,270),(270,409),(409,291)]



        FACEMESH_EYEBROW_LEFT = [
            (a, b) for a, b in [                
                (safe_index(index_mapping, 285), safe_index(index_mapping, 295)),
                (safe_index(index_mapping, 295), safe_index(index_mapping, 282)),
                (safe_index(index_mapping, 282), safe_index(index_mapping, 283)),
                (safe_index(index_mapping, 283), safe_index(index_mapping, 276)),

                (safe_index(index_mapping, 336), safe_index(index_mapping, 296)),
                (safe_index(index_mapping, 296), safe_index(index_mapping, 334)),
                (safe_index(index_mapping, 334), safe_index(index_mapping, 293)),
                (safe_index(index_mapping, 293), safe_index(index_mapping, 300)),

            ] if a is not None and b is not None
        ]


        FACEMESH_EYE_RIGHT = [
            (a, b) for a, b in [
                (safe_index(index_mapping, 144), safe_index(index_mapping, 145)),
                (safe_index(index_mapping, 145), safe_index(index_mapping, 153)),
                (safe_index(index_mapping, 153), safe_index(index_mapping, 154)),
                (safe_index(index_mapping, 154), safe_index(index_mapping, 155)),
                (safe_index(index_mapping, 155), safe_index(index_mapping, 133)),
                (safe_index(index_mapping, 133), safe_index(index_mapping, 173)),
                (safe_index(index_mapping, 173), safe_index(index_mapping, 157)),
                (safe_index(index_mapping, 157), safe_index(index_mapping, 158)),
                (safe_index(index_mapping, 158), safe_index(index_mapping, 159)),
                (safe_index(index_mapping, 159), safe_index(index_mapping, 160)),
                (safe_index(index_mapping, 160), safe_index(index_mapping, 161)),
                (safe_index(index_mapping, 161), safe_index(index_mapping, 246)),
                (safe_index(index_mapping, 246), safe_index(index_mapping, 33)),
                (safe_index(index_mapping, 33), safe_index(index_mapping, 7)),
                (safe_index(index_mapping, 7), safe_index(index_mapping, 163)),
                (safe_index(index_mapping, 163), safe_index(index_mapping, 144)),
                


            ] if a is not None and b is not None
        ]

        FACEMESH_EYEBROW_RIGHT = [
            (a, b) for a, b in [                
                (safe_index(index_mapping, 46), safe_index(index_mapping, 53)),
                (safe_index(index_mapping, 53), safe_index(index_mapping, 52)),
                (safe_index(index_mapping, 52), safe_index(index_mapping, 65)),
                (safe_index(index_mapping, 65), safe_index(index_mapping, 55)),

                (safe_index(index_mapping, 70), safe_index(index_mapping, 63)),
                (safe_index(index_mapping, 63), safe_index(index_mapping, 105)),
                (safe_index(index_mapping, 105), safe_index(index_mapping, 66)),
                (safe_index(index_mapping, 66), safe_index(index_mapping, 107)),

            ] if a is not None and b is not None
        ]


        FACE_LANDMARKS_LEFT_IRIS = [
            (a, b) for a, b in [                
                (safe_index(index_mapping, 469), safe_index(index_mapping, 470)),
                (safe_index(index_mapping, 470), safe_index(index_mapping, 471)),
                (safe_index(index_mapping, 471), safe_index(index_mapping, 472)),
                (safe_index(index_mapping, 472), safe_index(index_mapping, 469)),
            ] if a is not None and b is not None
        ]

        FACE_LANDMARKS_RIGHT_IRIS = [
            (a, b) for a, b in [                
                (safe_index(index_mapping, 474), safe_index(index_mapping, 475)),
                (safe_index(index_mapping, 475), safe_index(index_mapping, 476)),
                (safe_index(index_mapping, 476), safe_index(index_mapping, 477)),
                (safe_index(index_mapping, 477), safe_index(index_mapping, 474)),

            ] if a is not None and b is not None
        ]


        FACEMESH_CUSTOM_FACE_OVAL = [
            (a, b) for a, b in [
                (safe_index(index_mapping, 144), safe_index(index_mapping, 145)),
                (safe_index(index_mapping, 145), safe_index(index_mapping, 153)),
                (safe_index(index_mapping, 153), safe_index(index_mapping, 154)),
                (safe_index(index_mapping, 154), safe_index(index_mapping, 155)),
                (safe_index(index_mapping, 155), safe_index(index_mapping, 157)),
                (safe_index(index_mapping, 157), safe_index(index_mapping, 158)),
                (safe_index(index_mapping, 158), safe_index(index_mapping, 159)),
                (safe_index(index_mapping, 159), safe_index(index_mapping, 160)),
                (safe_index(index_mapping, 160), safe_index(index_mapping, 161)),
                (safe_index(index_mapping, 161), safe_index(index_mapping, 33)),
                (safe_index(index_mapping, 33), safe_index(index_mapping, 7)),
                (safe_index(index_mapping, 7), safe_index(index_mapping, 163)),

                (safe_index(index_mapping, 163), safe_index(index_mapping, 144)),
                (safe_index(index_mapping, 172), safe_index(index_mapping, 58)),
                (safe_index(index_mapping, 454), safe_index(index_mapping, 323)),
                (safe_index(index_mapping, 365), safe_index(index_mapping, 379)),
                (safe_index(index_mapping, 379), safe_index(index_mapping, 378)),
                (safe_index(index_mapping, 148), safe_index(index_mapping, 176)),
                (safe_index(index_mapping, 93), safe_index(index_mapping, 234)),
                (safe_index(index_mapping, 397), safe_index(index_mapping, 365)),
                (safe_index(index_mapping, 149), safe_index(index_mapping, 150)),
                (safe_index(index_mapping, 288), safe_index(index_mapping, 397)),
                (safe_index(index_mapping, 234), safe_index(index_mapping, 127)),
                (safe_index(index_mapping, 378), safe_index(index_mapping, 400)),
                (safe_index(index_mapping, 127), safe_index(index_mapping, 162)),
                (safe_index(index_mapping, 162), safe_index(index_mapping, 21))
            ] if a is not None and b is not None
        ]

        # import pdb;pdb.set_trace()



        # mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
        face_connection_spec = {}
        # if self.forehead_edge:
        #     for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
        #         face_connection_spec[edge] = head_draw
        # else:
        for edge in FACEMESH_CUSTOM_FACE_OVAL:
            face_connection_spec[edge] = head_draw
        # for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
        #     face_connection_spec[edge] = left_eye_draw
        # for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
        #     face_connection_spec[edge] = left_eyebrow_draw
        # for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
        #     face_connection_spec[edge] = right_eye_draw
        # for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
        #     face_connection_spec[edge] = right_eyebrow_draw
        # if iris_edge:
        #     for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
        #        face_connection_spec[edge] = left_iris_draw
        #     for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
        #        face_connection_spec[edge] = right_iris_draw
        # for edge in mp_face_mesh.FACEMESH_LIPS:
        #     face_connection_spec[edge] = mouth_draw
        

        for edge in FACEMESH_EYE_LEFT:
            face_connection_spec[edge] = left_eye_draw

        for edge in FACEMESH_EYEBROW_LEFT:
            face_connection_spec[edge] = left_eyebrow_draw

        for edge in FACEMESH_EYE_RIGHT:
            face_connection_spec[edge] = right_eye_draw

        for edge in FACEMESH_EYEBROW_RIGHT:
            face_connection_spec[edge] = right_eyebrow_draw

        for edge in FACE_LANDMARKS_LEFT_IRIS:
            face_connection_spec[edge] = left_iris_draw

        for edge in FACE_LANDMARKS_RIGHT_IRIS:
            face_connection_spec[edge] = right_iris_draw

        for edge in FACEMESH_LIPS_OUTER_BOTTOM_LEFT:
            face_connection_spec[edge] = mouth_draw_obl
        for edge in FACEMESH_LIPS_OUTER_BOTTOM_RIGHT:
            face_connection_spec[edge] = mouth_draw_obr
        for edge in FACEMESH_LIPS_INNER_BOTTOM_LEFT:
            face_connection_spec[edge] = mouth_draw_ibl
        for edge in FACEMESH_LIPS_INNER_BOTTOM_RIGHT:
            face_connection_spec[edge] = mouth_draw_ibr
        for edge in FACEMESH_LIPS_OUTER_TOP_LEFT:
            face_connection_spec[edge] = mouth_draw_otl
        for edge in FACEMESH_LIPS_OUTER_TOP_RIGHT:
            face_connection_spec[edge] = mouth_draw_otr
        for edge in FACEMESH_LIPS_INNER_TOP_LEFT:
            face_connection_spec[edge] = mouth_draw_itl
        for edge in FACEMESH_LIPS_INNER_TOP_RIGHT:
            face_connection_spec[edge] = mouth_draw_itr

        self.iris_point = iris_point
        
        self.face_connection_spec = face_connection_spec

    def draw_pupils(self, image, landmark_list, drawing_spec, halfwidth: int = 2):
        """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
        landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
        if len(image.shape) != 3:
            raise ValueError("Input image must be H,W,C.")
        image_rows, image_cols, image_channels = image.shape
        if image_channels != 3:  # BGR channels
            raise ValueError('Input image must contain three channel bgr data.')
        for idx, landmark in enumerate(landmark_list.landmark):
            if (
                    (landmark.HasField('visibility') and landmark.visibility < 0.9) or
                    (landmark.HasField('presence') and landmark.presence < 0.5)
            ):
                continue
            if landmark.x >= 1.0 or landmark.x < 0 or landmark.y >= 1.0 or landmark.y < 0:
                continue
            image_x = int(image_cols*landmark.x)
            image_y = int(image_rows*landmark.y)
            draw_color = None
            if isinstance(drawing_spec, Mapping):
                if drawing_spec.get(idx) is None:
                    continue
                else:
                    draw_color = drawing_spec[idx].color
            elif isinstance(drawing_spec, DrawingSpec):
                draw_color = drawing_spec.color
            image[image_y-halfwidth:image_y+halfwidth, image_x-halfwidth:image_x+halfwidth, :] = draw_color
    
    def draw_iris_points(self, image, point_list, halfwidth=2, normed=False):
        color = (255, 0, 0)
        for idx, point in enumerate(point_list):
            if normed:
                x, y = int(point[0] * image.shape[1]), int(point[1] * image.shape[0])
            else:
                x, y = int(point[0]), int(point[1])
            image[y-halfwidth:y+halfwidth, x-halfwidth:x+halfwidth, :] = color
        return image

    def draw_landmarks(self, image_size, keypoints, normed=False):
        ini_size = image_size #[512, 512]
        image = np.zeros([ini_size[1], ini_size[0], 3], dtype=np.uint8)
        new_landmarks = landmark_pb2.NormalizedLandmarkList()
        for i in range(keypoints.shape[0]):
            landmark = new_landmarks.landmark.add()
            if normed:
                landmark.x = keypoints[i, 0]
                landmark.y = keypoints[i, 1]
            else:
                landmark.x = keypoints[i, 0] / image_size[0]
                landmark.y = keypoints[i, 1] / image_size[1]
            landmark.z = 1.0

        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=new_landmarks,
            connections=self.face_connection_spec.keys(),
            landmark_drawing_spec=None,
            connection_drawing_spec=self.face_connection_spec
        )
        
        if self.iris_point:
            image = self.draw_iris_points(image, [keypoints[468], keypoints[473]], halfwidth=3, normed=normed)
        
        return image
    
    def draw_mask(self, image_size, keypoints, normed=False):
        mask = np.zeros([image_size[1], image_size[0], 3], dtype=np.uint8)
        if normed:
            keypoints[:, 0] *= image_size[0]
            keypoints[:, 1] *= image_size[1]
        
        head_idxs = [21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389]
        head_points = np.array(keypoints[head_idxs, :2], np.int32)

        mask = cv2.fillPoly(mask, [head_points], (255, 255, 255))
        mask = np.array(mask) / 255.0

        return mask