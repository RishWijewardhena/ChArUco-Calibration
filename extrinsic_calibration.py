"""
Extrinsic Calibration Module

Handles camera extrinsic calibration (pose estimation relative to ChArUco board).
Computes rotation and translation vectors.
"""

import cv2
import numpy as np
import json
from config import CALIB_FILE, EXTRINSICS_FILE


class ExtrinsicCalibration:
    """Manages extrinsic calibration workflow"""
    
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        """
        Initialize extrinsic calibration.
        
        Args:
            camera_matrix (np.ndarray, optional): Camera intrinsic matrix
            dist_coeffs (np.ndarray, optional): Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvec = None
        self.tvec = None
    
    def set_intrinsics(self, camera_matrix, dist_coeffs):
        """
        Set camera intrinsic parameters.
        
        Args:
            camera_matrix (np.ndarray): Camera intrinsic matrix
            dist_coeffs (np.ndarray): Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def capture_pose(self, charuco_corners, charuco_ids, board):
        """
        Compute camera pose (rvec, tvec) from detected ChArUco board.
        
        Args:
            charuco_corners: Detected ChArUco corners
            charuco_ids: Detected ChArUco IDs
            board: ChArUco board instance
        
        Returns:
            tuple: (success, message, rvec, tvec)
        
        Raises:
            ValueError: If intrinsics not set or insufficient corners detected
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera intrinsics not set. Load calibration first.")
        
        if charuco_corners is None or len(charuco_corners) < 4:
            return False, "Need at least 4 ChArUco corners for pose estimation", None, None
        
        try:
            # Get 3D object points for detected corners
            obj_points = board.getChessboardCorners()
            obj_pts = np.array([obj_points[id[0]] for id in charuco_ids], dtype=np.float32)
            img_pts = charuco_corners.reshape(-1, 2).astype(np.float32)
            
            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, 
                self.camera_matrix, 
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                self.rvec = rvec
                self.tvec = tvec
                return True, "Extrinsics captured successfully", rvec, tvec
            else:
                return False, "solvePnP failed", None, None
                
        except Exception as e:
            return False, f"Pose estimation error: {str(e)}", None, None
    
    def save_extrinsics(self, rvec=None, tvec=None, filepath=None):
        """
        Save extrinsic parameters to JSON file.
        
        Args:
            rvec (np.ndarray, optional): Rotation vector. Uses self.rvec if None
            tvec (np.ndarray, optional): Translation vector. Uses self.tvec if None
            filepath (str, optional): Output file path. Defaults to EXTRINSICS_FILE
        
        Returns:
            str: Path to saved file
        
        Raises:
            ValueError: If rvec/tvec not provided and not set
        """
        if rvec is None:
            rvec = self.rvec
        if tvec is None:
            tvec = self.tvec
        
        if rvec is None or tvec is None:
            raise ValueError("Rotation or translation vector not available")
        
        if filepath is None:
            filepath = EXTRINSICS_FILE
        
        extrinsics_data = {
            'rvec': rvec.flatten().tolist(),
            'tvec': tvec.flatten().tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(extrinsics_data, f, indent=2)
        
        return filepath
    
    @staticmethod
    def load_intrinsics(filepath=None):
        """
        Load camera intrinsics from calibration file.
        
        Args:
            filepath (str, optional): Calibration file path. Defaults to CALIB_FILE
        
        Returns:
            tuple: (camera_matrix, dist_coeffs) as numpy arrays
        
        Raises:
            FileNotFoundError: If calibration file doesn't exist
        """
        if filepath is None:
            filepath = CALIB_FILE
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        camera_matrix = np.array(data['camera_matrix'], dtype=np.float64)
        dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float64)
        
        return camera_matrix, dist_coeffs
    
    @staticmethod
    def load_extrinsics(filepath=None):
        """
        Load extrinsic parameters from file.
        
        Args:
            filepath (str, optional): Extrinsics file path. Defaults to EXTRINSICS_FILE
        
        Returns:
            tuple: (rvec, tvec) as numpy arrays
        
        Raises:
            FileNotFoundError: If extrinsics file doesn't exist
        """
        if filepath is None:
            filepath = EXTRINSICS_FILE
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        rvec = np.array(data['rvec'], dtype=np.float64).reshape(3, 1)
        tvec = np.array(data['tvec'], dtype=np.float64).reshape(3, 1)
        
        return rvec, tvec
