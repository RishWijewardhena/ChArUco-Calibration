"""
Configuration Module for ChArUco Calibration Application

Contains all configuration constants and detector initialization functions.
"""

import cv2

# ==================== FILE PATHS ====================
CALIB_FILE = "camera_calibration.json"
EXTRINSICS_FILE = "camera_extrinsics.json"

# ==================== CHARUCO BOARD PARAMETERS ====================
DICT_TYPE = cv2.aruco.DICT_4X4_50
SQUARES_X = 6  # number of squares in X direction
SQUARES_Y = 5  # number of squares in Y direction
SQUARE_LENGTH = 0.010  # meters (adjust as needed)
MARKER_LENGTH = 0.008   # meters (adjust as needed)
MIN_CHARUCO_CORNERS = 4



# color inversion
INVERT_COLORS = True 

# ==================== DETECTOR INITIALIZATION ====================

def get_aruco_board():
    """
    Create and return the ChArUco board object.
    
    Returns:
        cv2.aruco.CharucoBoard: Configured ChArUco board
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), 
        SQUARE_LENGTH, 
        MARKER_LENGTH, 
        aruco_dict
    )
    return board


def get_detectors():
    """
    Initialize and return ArUco and ChArUco detectors.
    
    Returns:
        tuple: (charuco_detector, aruco_detector, board, aruco_dict)
    
    Raises:
        RuntimeError: If detector initialization fails
    """
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y), 
            SQUARE_LENGTH, 
            MARKER_LENGTH, 
            aruco_dict
        )
        
        params = cv2.aruco.DetectorParameters()
        charuco_params = cv2.aruco.CharucoParameters()
        
        charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, params)
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        
        return charuco_detector, aruco_detector, board, aruco_dict
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize detectors: {str(e)}")
