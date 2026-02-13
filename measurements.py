"""
Measurement Module

Handles 3D distance measurements from 2D image points using camera calibration data.
Projects image points to a measurement plane and computes real-world distances.
"""

import cv2
import numpy as np
import json
from config import CALIB_FILE, EXTRINSICS_FILE


class Measurement:
    """Manages measurement workflow with calibrated camera"""
    
    def __init__(self, camera_matrix=None, dist_coeffs=None, rvec=None, tvec=None):
        """
        Initialize measurement system.
        
        Args:
            camera_matrix (np.ndarray, optional): Camera intrinsic matrix
            dist_coeffs (np.ndarray, optional): Distortion coefficients
            rvec (np.ndarray, optional): Rotation vector (extrinsics)
            tvec (np.ndarray, optional): Translation vector (extrinsics)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvec = rvec
        self.tvec = tvec
        self.click_points = []
    
    def set_calibration(self, camera_matrix, dist_coeffs, rvec, tvec):
        """
        Set all calibration parameters.
        
        Args:
            camera_matrix (np.ndarray): Camera intrinsic matrix
            dist_coeffs (np.ndarray): Distortion coefficients
            rvec (np.ndarray): Rotation vector
            tvec (np.ndarray): Translation vector
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvec = rvec
        self.tvec = tvec
    
    def undistort_frame(self, frame):
        """
        Apply lens distortion correction to a frame.
        
        Args:
            frame (np.ndarray): Input image
        
        Returns:
            np.ndarray: Undistorted image
        
        Raises:
            ValueError: If camera calibration not set
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera calibration not set")
        
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
    
    def add_click_point(self, pixel_coords):
        """
        Add a clicked point for measurement.
        
        Args:
            pixel_coords (tuple): (x, y) pixel coordinates
        
        Returns:
            int: Number of points stored
        """
        self.click_points.append(pixel_coords)
        return len(self.click_points)
    
    def get_click_count(self):
        """
        Get number of clicked points.
        
        Returns:
            int: Number of points
        """
        return len(self.click_points)
    
    def reset_points(self):
        """Clear all clicked points"""
        self.click_points = []
    
    def compute_distance(self):
        """
        Compute 3D distance between two clicked points.
        
        Returns:
            tuple: (success, message, distance_m, distance_cm, point1_3d, point2_3d)
        
        Raises:
            ValueError: If calibration not complete or insufficient points
        """
        if len(self.click_points) < 2:
            return False, "Need 2 points for distance measurement", None, None, None, None
        
        if self.camera_matrix is None or self.dist_coeffs is None:
            return False, "Camera intrinsics not loaded", None, None, None, None
        
        if self.rvec is None or self.tvec is None:
            return False, "Camera extrinsics not loaded", None, None, None, None
        
        try:
            # Get rotation matrix from rvec
            R, _ = cv2.Rodrigues(self.rvec)
            
            # Plane normal in camera frame (Z-axis of board frame)
            plane_normal_cam = R[:, 2]
            
            # Distance from origin to plane along normal
            d = -plane_normal_cam.dot(self.tvec.flatten())
            
            # Project both points to the measurement plane
            pt1_3d = self.image_to_plane(self.click_points[0], R, plane_normal_cam, d)
            pt2_3d = self.image_to_plane(self.click_points[1], R, plane_normal_cam, d)
            
            # Compute Euclidean distance
            distance_m = np.linalg.norm(pt2_3d - pt1_3d)
            distance_cm = distance_m * 100
            
            return True, "Distance computed successfully", distance_m, distance_cm, pt1_3d, pt2_3d
            
        except Exception as e:
            return False, f"Error computing distance: {str(e)}", None, None, None, None
    
    def image_to_plane(self, pt, R, plane_normal_cam, d):
        """
        Project image point to measurement plane using ray-plane intersection.
        
        Args:
            pt (tuple): (x, y) pixel coordinates
            R (np.ndarray): Rotation matrix (3x3)
            plane_normal_cam (np.ndarray): Plane normal in camera frame (3,)
            d (float): Distance from origin to plane
        
        Returns:
            np.ndarray: 3D point on plane (x, y, z) in board coordinates
        
        Raises:
            RuntimeError: If ray is parallel to plane
        """
        # Points come from the UNDISTORTED image, so we only need to apply
        # the inverse camera matrix to get normalized ray direction.
        # Do NOT call cv2.undistortPoints() here — that would double-undistort
        # since the frozen frame the user clicks on is already undistorted.
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        x = (pt[0] - cx) / fx
        y = (pt[1] - cy) / fy
        ray_cam = np.array([x, y, 1.0])
        
        denom = plane_normal_cam.dot(ray_cam)
        if abs(denom) < 1e-9:
            raise RuntimeError('Ray nearly parallel to plane')
        
        s = -d / denom
        Xc = s * ray_cam
        obj_xy = R[:, :2].T.dot(Xc - self.tvec.flatten())
        
        return np.array([obj_xy[0], obj_xy[1], 0.0])
    
    @staticmethod
    def load_calibrations(calib_path=None, extrin_path=None):
        """
        Load both intrinsic and extrinsic calibrations from files.
        
        Args:
            calib_path (str, optional): Intrinsics file path. Defaults to CALIB_FILE
            extrin_path (str, optional): Extrinsics file path. Defaults to EXTRINSICS_FILE
        
        Returns:
            tuple: (camera_matrix, dist_coeffs, rvec, tvec) as numpy arrays
        
        Raises:
            FileNotFoundError: If calibration files don't exist
        """
        if calib_path is None:
            calib_path = CALIB_FILE
        if extrin_path is None:
            extrin_path = EXTRINSICS_FILE
        
        # Load intrinsics
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        
        camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float64)
        dist_coeffs = np.array(calib_data['dist_coeffs'], dtype=np.float64)
        
        # Load extrinsics
        with open(extrin_path, 'r') as f:
            extrin_data = json.load(f)
        
        rvec = np.array(extrin_data['rvec'], dtype=np.float64).reshape(3, 1)
        tvec = np.array(extrin_data['tvec'], dtype=np.float64).reshape(3, 1)
        
        return camera_matrix, dist_coeffs, rvec, tvec
