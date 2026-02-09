"""
Intrinsic Calibration Module

Handles camera intrinsic calibration using ChArUco boards.
Computes camera matrix and distortion coefficients.
"""

import cv2
import numpy as np
import json
from PyQt6.QtCore import QThread, pyqtSignal
from config import CALIB_FILE


class CalibrationWorker(QThread):
    """Background thread for running calibration calculations"""
    finished = pyqtSignal(bool, str, object)
    progress = pyqtSignal(str)
    
    def __init__(self, corners, ids, board, image_size):
        """
        Initialize calibration worker.
        
        Args:
            corners (list): List of detected ChArUco corner arrays
            ids (list): List of detected ChArUco ID arrays
            board: ChArUco board instance
            image_size (tuple): Image dimensions (width, height)
        """
        super().__init__()
        self.corners = corners
        self.ids = ids
        self.board = board
        self.image_size = image_size
        
    def run(self):
        """Execute calibration algorithm"""
        try:
            self.progress.emit("Running calibration algorithm...")
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                self.corners, self.ids, self.board, self.image_size, None, None
            )
            
            if not ret or ret > 2.0:  # RMS error threshold
                self.finished.emit(False, "Calibration RMS error too high. Collect better frames.", None)
                return
                
            calib_data = {
                'camera_matrix': camera_matrix.tolist(),
                'dist_coeffs': dist_coeffs.tolist(),
                'rms': float(ret),
                'image_size': self.image_size
            }
            
            self.progress.emit(f"Calibration successful! RMS = {ret:.4f}")
            self.finished.emit(True, f"Calibration completed with RMS error: {ret:.4f}", calib_data)
            
        except Exception as e:
            self.finished.emit(False, f"Calibration failed: {str(e)}", None)


class IntrinsicCalibration:
    """Manages intrinsic calibration workflow and data storage"""
    
    def __init__(self):
        """Initialize calibration data storage"""
        self.all_charuco_corners = []
        self.all_charuco_ids = []
        self.image_size = None
        
    def add_frame(self, charuco_corners, charuco_ids, image_size):
        """
        Add a captured frame to the calibration dataset.
        
        Args:
            charuco_corners: Detected ChArUco corners
            charuco_ids: Detected ChArUco IDs
            image_size (tuple): Image dimensions (width, height)
        
        Returns:
            int: Total number of frames captured
        """
        self.all_charuco_corners.append(charuco_corners)
        self.all_charuco_ids.append(charuco_ids)
        
        if self.image_size is None:
            self.image_size = image_size
        
        return len(self.all_charuco_corners)
    
    def get_frame_count(self):
        """
        Get the number of captured calibration frames.
        
        Returns:
            int: Number of frames
        """
        return len(self.all_charuco_corners)
    
    def clear_frames(self):
        """Clear all captured calibration frames"""
        self.all_charuco_corners = []
        self.all_charuco_ids = []
        self.image_size = None
    
    def run_calibration(self, board):
        """
        Create and return a CalibrationWorker thread to run calibration.
        
        Args:
            board: ChArUco board instance
        
        Returns:
            CalibrationWorker: Thread ready to start
        
        Raises:
            ValueError: If insufficient frames collected
        """
        if self.get_frame_count() < 8:
            raise ValueError("Need at least 8 frames for calibration")
        
        return CalibrationWorker(
            self.all_charuco_corners, 
            self.all_charuco_ids, 
            board, 
            self.image_size
        )
    
    @staticmethod
    def save_calibration(calib_data, filepath=None):
        """
        Save calibration data to JSON file.
        
        Args:
            calib_data (dict): Calibration data containing camera_matrix, dist_coeffs, rms, image_size
            filepath (str, optional): Output file path. Defaults to CALIB_FILE from config
        
        Returns:
            str: Path to saved file
        """
        if filepath is None:
            filepath = CALIB_FILE
        
        with open(filepath, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        return filepath
    
    @staticmethod
    def load_calibration(filepath=None):
        """
        Load calibration data from JSON file.
        
        Args:
            filepath (str, optional): Input file path. Defaults to CALIB_FILE from config
        
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
