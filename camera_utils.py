"""
Camera Utilities Module - OS-Aware Camera Handling

Provides cross-platform camera initialization for Windows, Linux, and macOS.
Supports environment variable override for custom camera sources.
"""

import platform
import os
import cv2


def get_camera_source():
    """
    Get the appropriate camera source based on the operating system.
    
    OS-specific defaults:
        - Windows: 0 (first webcam index)
        - Linux: "/dev/video0" (V4L2 device path)
        - macOS: 0 (first webcam index)
    
    Environment variable override:
        Set CAMERA_SOURCE to override the default (e.g., CAMERA_SOURCE=1 or CAMERA_SOURCE=/dev/video2)
    
    Returns:
        int or str: Camera source (integer index for Windows/Mac, device path for Linux)
    """
    system = platform.system()

    # Choose camera source based on OS
    if system == "Windows":
        cam_source = 0  # usually first webcam
    elif system == "Darwin":  # macOS
        cam_source = 0
    else:  # Linux and others
        cam_source = "/dev/video0"  # Linux default camera

    # Optional: override via environment variable
    cam_source = os.getenv("CAMERA_SOURCE", cam_source)

    # Convert to int if possible (needed for Windows/Mac)
    try:
        cam_source = int(cam_source)
    except (ValueError, TypeError):
        pass

    return cam_source


def get_camera_backend():
    """
    Get the appropriate OpenCV camera backend based on the operating system.
    
    OS-specific backends:
        - Windows: cv2.CAP_DSHOW (DirectShow)
        - Linux: cv2.CAP_V4L2 (Video4Linux2)
        - macOS: cv2.CAP_AVFOUNDATION (AVFoundation)
    
    Returns:
        int: OpenCV VideoCapture backend constant
    """
    system = platform.system()
    
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Darwin":  # macOS
        return cv2.CAP_AVFOUNDATION
    else:  # Linux
        return cv2.CAP_V4L2


def open_camera(cam_source=None, backend=None):
    """
    Open camera with OS-appropriate settings.
    
    Args:
        cam_source (int or str, optional): Camera source. If None, uses get_camera_source()
        backend (int, optional): OpenCV backend. If None, uses get_camera_backend()
    
    Returns:
        cv2.VideoCapture: Opened camera capture object
    
    Raises:
        RuntimeError: If camera cannot be opened
    """
    if cam_source is None:
        cam_source = get_camera_source()
    
    if backend is None:
        backend = get_camera_backend()
    
    cap = cv2.VideoCapture(cam_source, backend)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_source} with backend {backend}")
    
    return cap


def get_available_cameras(max_test=10):
    """
    Detect available camera indices on the system.
    
    Args:
        max_test (int): Maximum number of camera indices to test
    
    Returns:
        list: List of available camera indices
    """
    available_cameras = []
    backend = get_camera_backend()
    
    for i in range(max_test):
        try:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        except:
            continue
    
    return available_cameras
