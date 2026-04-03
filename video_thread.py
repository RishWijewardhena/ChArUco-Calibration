"""
Video Thread Module - Background Video Capture and Processing

Provides QThread-based video capture with ChArUco detection for real-time display.
"""

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
import camera_utils
from config import INVERT_COLORS
import time


class VideoThread(QThread):
    """Thread for continuous video capture and processing"""
    change_pixmap = pyqtSignal(QImage)
    detection_info = pyqtSignal(dict)
    camera_lost = pyqtSignal(str)

    def __init__(self, camera_source, detector, board, invert_colors=None, target_fps=30):
        """
        Initialize video capture thread.

        Args:
            camera_source (int or str): Camera source from camera_utils.get_camera_source()
            detector: ChArUco detector instance
            board: ChArUco board instance
            invert_colors (bool, optional): Whether to invert colors before detection.
                                       If None, uses INVERT_COLORS from config
            target_fps (int): Target frames per second to prevent UI lag
        """
        super().__init__()
        self.camera_source = camera_source
        self.detector = detector
        self.board = board
        self.running = True
        self.cap = None
        self.invert_colors = invert_colors if invert_colors is not None else INVERT_COLORS
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.max_read_failures = 20
        
    def run(self):
        """Main thread execution loop - captures and processes frames"""
        try:
            # Open camera with OS-appropriate backend
            self.cap = camera_utils.open_camera(self.camera_source)

            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_source}")
            
            # Set camera resolution to 1280×960 for better calibration accuracy
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            
            last_frame_time = time.time()
            read_failures = 0

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    read_failures += 1

                    # Camera unplug/disconnect often causes repeated read failures.
                    if read_failures >= self.max_read_failures:
                        self.camera_lost.emit("Camera stream lost. The camera may be disconnected.")
                        break

                    time.sleep(0.05)
                    continue

                read_failures = 0

                # Frame rate limiting - skip processing if too soon
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < self.frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent CPU spinning
                    continue
                last_frame_time = current_time

                # Process frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.invert_colors:
                    gray = cv2.bitwise_not(gray)

                charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)

                # Draw detections
                display_frame = frame.copy()
                if marker_ids is not None and len(marker_ids) > 0:
                    cv2.aruco.drawDetectedMarkers(display_frame, marker_corners, marker_ids)
                if charuco_ids is not None and len(charuco_ids) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids)

                # Emit detection info
                info = {
                    'corners_detected': len(charuco_ids) if charuco_ids is not None else 0,
                    'charuco_corners': charuco_corners,
                    'charuco_ids': charuco_ids,
                    'frame': frame
                }
                self.detection_info.emit(info)

                # Convert to Qt format
                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.change_pixmap.emit(qt_image)
                
        except Exception as e:
            print(f"Video thread error: {e}")
            self.camera_lost.emit(f"Camera error: {e}")
        finally:
            if self.cap:
                self.cap.release()
    
    def stop(self):
        """Stop the video thread and release camera"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.quit()
        if not self.wait(1500):
            self.terminate()
            self.wait(500)
