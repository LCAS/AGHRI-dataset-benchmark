import numpy as np
import cv2


class Proyector:

    def __init__(self, P, R0, V2C, w, h):
        self.w = w
        self.h = h
        R0_padded = np.column_stack([np.vstack([R0, [0, 0, 0]]), [0, 0, 0, 1]])
        V2C_padded = np.vstack((V2C, [0, 0, 0, 1]))
        self.trafo_matrix = np.dot(P, np.dot(R0_padded, V2C_padded))


    def proyect(self, bboxes_3d):

        bboxes_3d_numpy = self._markerarray2np(bboxes_3d)
        proyected_3d_bboxes = self._proyect_3d_bboxes(bboxes_3d_numpy)

        if proyected_3d_bboxes.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32)

        bboxes_2d = self._get_2dbboxes(proyected_3d_bboxes)

        return bboxes_2d

    def _get_2dbboxes(self, proyected_bboxes: np.ndarray) -> np.ndarray:
        """
        args:
            proyected_bboxes (np.ndarray): (m, 8, 2) array with projected 3D bounding boxes
        returns:
            np.ndarray: (m, 4) array with 2D bounding boxes [minx, miny, maxx, maxy]
        """
        min_xy = proyected_bboxes.min(axis=1)  # (m, 2)
        max_xy = proyected_bboxes.max(axis=1)  # (m, 2)

        # Concatenate into (m, 4): [minx, miny, maxx, maxy]
        bboxes_2d = np.concatenate([min_xy, max_xy], axis=1)  # (m, 4)

        return bboxes_2d.astype(np.float32)

    def _proyect_3d_bboxes(self, bboxes_3d_numpy):
        '''converts the real world 3d bboxes into proyected 3d bboxes

        args:
            bboxes_3d_numpy -> np.array.shape == [M, 8, 3] (meters)
        return: 
            proyected_3d_bboxes -> np.array.shape == [M, 8, 2] (pixels)'''

        n = bboxes_3d_numpy.shape[0]
        projected_boxes = []

        for i in range(n):
            corners_3d = bboxes_3d_numpy[i]  # (8, 3)
            # Añadir coordenada homogénea
            corners_hom = np.hstack([corners_3d, np.ones((8, 1))])  # (8, 4)
            # Proyección
            pts_2d_homo = (self.trafo_matrix @ corners_hom.T).T  # (8, 3)
            pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]  # Normalizar por Z

            # Verificar que todos los puntos están dentro de la imagen
            inside = (
                (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < self.w) &
                (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < self.h)
            )
            if np.all(inside):
                projected_boxes.append(pts_2d)

        return np.array(projected_boxes, dtype=np.float32) 

    def _markerarray2np(self, marker_array):
        """
        Args:
            marker_array (visualization_msgs.msg.MarkerArray): Input marker array
            
        Returns:
            numpy.ndarray: Array of shape [n, 8, 3] where:
                          - n = number of objects (markers)
                          - 8 = 8 corners per bounding box
                          - 3 = (x, y, z) coordinates
                          
        Corner ordering (following standard convention):
            Bottom face (z_min):     Top face (z_max):
            3 -------- 2            7 -------- 6
            |          |            |          |
            |          |            |          |
            0 -------- 1            4 -------- 5
            
        Coordinate system: ROS standard (x=forward, y=left, z=up)
        """

        marker_array = marker_array.markers
        corners_list = []
        
        for i, marker in enumerate(marker_array.markers):
                
            # Extract pose information
            pos_x = marker.pose.position.x
            pos_y = marker.pose.position.y
            pos_z = marker.pose.position.z
            
            # Extract quaternion orientation
            qx = marker.pose.orientation.x
            qy = marker.pose.orientation.y
            qz = marker.pose.orientation.z
            qw = marker.pose.orientation.w
            
            # Extract scale (dimensions)
            length = marker.scale.x  # x-direction (forward)
            width = marker.scale.y   # y-direction (left)
            height = marker.scale.z  # z-direction (up)
            
            # Calculate 8 corners of the bounding box
            corners = self._calculate_bbox_corners(
                pos_x, pos_y, pos_z,
                qx, qy, qz, qw,
                length, width, height
            )
            
            corners_list.append(corners)
        
        # Convert to numpy array [n_objects, 8_corners, 3_coordinates]
        corners_array = np.array(corners_list)
        
        return corners_array


    def _calculate_bbox_corners(self, center_x, center_y, center_z, qx, qy, qz, qw, length, width, height):
        """
        Args:
            center_x, center_y, center_z (float): Center position of bounding box
            qx, qy, qz, qw (float): Quaternion orientation
            length, width, height (float): Dimensions of bounding box
            
        Returns:
            numpy.ndarray: Array of shape [8, 3] containing corner coordinates
        """
        # Half dimensions
        l_2 = length / 2.0
        w_2 = width / 2.0
        h_2 = height / 2.0
        
        # Define 8 corners in local coordinate system (before rotation)
        # Origin at center, following ROS convention (x=forward, y=left, z=up)
        local_corners = np.array([
            [-l_2, -w_2, -h_2],  # 0: back-right-bottom
            [+l_2, -w_2, -h_2],  # 1: front-right-bottom
            [+l_2, +w_2, -h_2],  # 2: front-left-bottom
            [-l_2, +w_2, -h_2],  # 3: back-left-bottom
            [-l_2, -w_2, +h_2],  # 4: back-right-top
            [+l_2, -w_2, +h_2],  # 5: front-right-top
            [+l_2, +w_2, +h_2],  # 6: front-left-top
            [-l_2, +w_2, +h_2],  # 7: back-left-top
        ])
        
        # Convert quaternion to rotation matrix
        rotation_matrix = self._quaternion_to_rotation_matrix(qx, qy, qz, qw)
        
        # Apply rotation to all corners
        rotated_corners = np.dot(local_corners, rotation_matrix.T)
        
        # Translate to world coordinates
        center = np.array([center_x, center_y, center_z])
        world_corners = rotated_corners + center
        
        return world_corners

    def _quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """
        Args:
            qx, qy, qz, qw (float): Quaternion components
            
        Returns:
            numpy.ndarray: 3x3 rotation matrix
        """
        # Normalize quaternion
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm == 0:
            return np.eye(3)
        
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # Convert to rotation matrix using standard formula
        r11 = 1 - 2*(qy*qy + qz*qz)
        r12 = 2*(qx*qy - qz*qw)
        r13 = 2*(qx*qz + qy*qw)
        
        r21 = 2*(qx*qy + qz*qw)
        r22 = 1 - 2*(qx*qx + qz*qz)
        r23 = 2*(qy*qz - qx*qw)
        
        r31 = 2*(qx*qz - qy*qw)
        r32 = 2*(qy*qz + qx*qw)
        r33 = 1 - 2*(qx*qx + qy*qy)
        
        rotation_matrix = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])
        
        return rotation_matrix
