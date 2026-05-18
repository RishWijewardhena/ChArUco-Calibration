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
    frame_timing = pyqtSignal(dict)  # Emit timing diagnostics: {'avg_process_time_ms': float, 'dropped_frames': int}

    def __init__(
        self,
        camera_source,
        detector,
        board,
        invert_colors=None,
        target_fps=30,
        frame_width=1280,
        frame_height=960,
    ):
        """
        Initialize video capture thread.

        Args:
            camera_source (int or str): Camera source from camera_utils.get_camera_source()
            detector: ChArUco detector instance
            board: ChArUco board instance
            invert_colors (bool, optional): Whether to invert colors before detection.
                                       If None, uses INVERT_COLORS from config
            target_fps (int): Target frames per second to prevent UI lag
            frame_width (int): Desired camera frame width
            frame_height (int): Desired camera frame height
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
        self.frame_width = max(1, int(frame_width))
        self.frame_height = max(1, int(frame_height))
        
        # Frame-skipping backpressure: track if UI is still rendering previous frame
        self.rendering = False
        self.dropped_frame_count = 0
        self.frame_times = []  # Rolling window of last 30 frame times for averaging
        
    def run(self):
        """Main thread execution loop - captures and processes frames"""
        try:
            # Open camera with OS-appropriate backend
            self.cap = camera_utils.open_camera(self.camera_source)

            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_source}")
            
            # Apply user-selected frame size.
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            last_frame_time = time.time()
            read_failures = 0
            last_diagnostic_time = time.time()

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
                frame_process_start = time.time()

                # Process frame (FULL RESOLUTION - detection must match calibration matrix)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.invert_colors:
                    gray = cv2.bitwise_not(gray)

                charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)

                # Draw detections (on full resolution frame)
                display_frame = frame.copy()
                if marker_ids is not None and len(marker_ids) > 0:
                    cv2.aruco.drawDetectedMarkers(display_frame, marker_corners, marker_ids)
                if charuco_ids is not None and len(charuco_ids) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids)

                # Emit detection info (always at full resolution for calibration accuracy)
                info = {
                    'corners_detected': len(charuco_ids) if charuco_ids is not None else 0,
                    'charuco_corners': charuco_corners,
                    'charuco_ids': charuco_ids,
                    'frame': frame
                }
                self.detection_info.emit(info)

                # Frame-skipping backpressure: skip emission if UI is still rendering previous frame
                # This prevents frame queuing and reduces flashing at high resolution
                if self.rendering:
                    self.dropped_frame_count += 1
                else:
                    # Convert to Qt format for display
                    rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    self.rendering = True
                    self.change_pixmap.emit(qt_image)
                    self.rendering = False
                
                # Track frame processing time (milliseconds)
                frame_process_time = (time.time() - frame_process_start) * 1000
                self.frame_times.append(frame_process_time)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                
                # Emit diagnostics every ~1 second
                if (time.time() - last_diagnostic_time) >= 1.0:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
                    self.frame_timing.emit({
                        'avg_process_time_ms': avg_frame_time,
                        'dropped_frames': self.dropped_frame_count,
                        'frame_width': self.frame_width,
                        'frame_height': self.frame_height
                    })
                    self.dropped_frame_count = 0  # Reset counter after reporting
                    last_diagnostic_time = time.time()
                
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
