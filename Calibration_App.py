"""
ChArUco Camera Calibration & Measurement Tool - Desktop GUI
Professional PyQt6 interface for camera calibration and measurement using ChArUco boards.

Modular architecture with OS-aware camera handling.

Requirements:
    pip install PyQt6 opencv-contrib-python numpy

To create executable:
    pip install pyinstaller
    pyinstaller --onefile --windowed --name "ChArUco_Calibration" --icon "app.ico" Calibration_App.py

Author: Camera Calibration Tool
Version: 3.0 - Modular Architecture
"""

import sys
import cv2
import numpy as np
import os
import json
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QTabWidget, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QMessageBox, QFileDialog, QProgressBar, QFrame, QComboBox,
                             QCheckBox, QDialog, QDialogButtonBox, QFormLayout)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon

# Import our modules
from config import (
    CALIB_FILE,
    EXTRINSICS_FILE,
    MIN_CHARUCO_CORNERS,
    RUNTIME_SETTINGS_FILE,
    get_default_runtime_settings,
    validate_runtime_settings,
    get_aruco_dictionary_names,
    create_detectors,
)
from camera_utils import get_camera_source, get_available_cameras_with_names, get_camera_backend
from video_thread import VideoThread
from intrinsic_calibration import IntrinsicCalibration, CalibrationWorker
from extrinsic_calibration import ExtrinsicCalibration
from measurements import Measurement


class CameraParametersDialog(QDialog):
    """Popup window to collect runtime camera and board parameters."""

    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera and ChArUco Parameters")
        self.setModal(True)
        self.setMinimumWidth(420)

        settings = dict(get_default_runtime_settings())
        settings.update(current_settings or {})

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.dict_combo = QComboBox()
        for name in get_aruco_dictionary_names():
            self.dict_combo.addItem(name)
        current_dict = settings.get("dict_type", "")
        dict_index = self.dict_combo.findText(str(current_dict))
        if dict_index >= 0:
            self.dict_combo.setCurrentIndex(dict_index)
        form.addRow("Dictionary", self.dict_combo)

        self.squares_x_spin = QSpinBox()
        self.squares_x_spin.setRange(2, 50)
        self.squares_x_spin.setValue(int(settings["squares_x"]))
        form.addRow("Squares X", self.squares_x_spin)

        self.squares_y_spin = QSpinBox()
        self.squares_y_spin.setRange(2, 50)
        self.squares_y_spin.setValue(int(settings["squares_y"]))
        form.addRow("Squares Y", self.squares_y_spin)

        self.square_length_spin = QDoubleSpinBox()
        self.square_length_spin.setRange(0.0001, 10.0)
        self.square_length_spin.setDecimals(6)
        self.square_length_spin.setSingleStep(0.001)
        self.square_length_spin.setValue(float(settings["square_length"]))
        form.addRow("Square Length (m)", self.square_length_spin)

        self.marker_length_spin = QDoubleSpinBox()
        self.marker_length_spin.setRange(0.0001, 10.0)
        self.marker_length_spin.setDecimals(6)
        self.marker_length_spin.setSingleStep(0.001)
        self.marker_length_spin.setValue(float(settings["marker_length"]))
        form.addRow("Marker Length (m)", self.marker_length_spin)

        self.invert_checkbox = QCheckBox("Invert grayscale before detection")
        self.invert_checkbox.setChecked(bool(settings["invert_colors"]))
        form.addRow("Detection", self.invert_checkbox)

        self.frame_width_spin = QSpinBox()
        self.frame_width_spin.setRange(160, 8192)
        self.frame_width_spin.setSingleStep(16)
        self.frame_width_spin.setValue(int(settings["frame_width"]))
        form.addRow("Frame Width", self.frame_width_spin)

        self.frame_height_spin = QSpinBox()
        self.frame_height_spin.setRange(120, 4320)
        self.frame_height_spin.setSingleStep(16)
        self.frame_height_spin.setValue(int(settings["frame_height"]))
        form.addRow("Frame Height", self.frame_height_spin)

        layout.addLayout(form)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self._validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_settings(self):
        """Return dialog values as a runtime settings dictionary."""
        return {
            "dict_type": self.dict_combo.currentText(),
            "squares_x": self.squares_x_spin.value(),
            "squares_y": self.squares_y_spin.value(),
            "square_length": float(self.square_length_spin.value()),
            "marker_length": float(self.marker_length_spin.value()),
            "invert_colors": self.invert_checkbox.isChecked(),
            "frame_width": self.frame_width_spin.value(),
            "frame_height": self.frame_height_spin.value(),
        }

    def _validate_and_accept(self):
        """Validate input before closing the popup."""
        is_valid, message, _ = validate_runtime_settings(self.get_settings())
        if not is_valid:
            QMessageBox.warning(self, "Invalid Parameters", message)
            return
        self.accept()


class ChArUcoCalibrationGUI(QMainWindow):
    """Main GUI application for ChArUco calibration and measurement"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChArUco Camera Calibration & Measurement Tool")
        self.setGeometry(50, 50, 4000, 3000)

        self.runtime_settings = self.load_runtime_settings()
        
        # Initialize detector components
        self.init_detectors()
        
        # Get camera source based on OS
        self.camera_source = get_camera_source()

        self.setWindowIcon(QIcon(self.resource_path("App_icon.png")))
        
        # Initialize workflow modules
        self.intrinsic = IntrinsicCalibration()
        self.extrinsic = ExtrinsicCalibration()
        self.measurement = Measurement()
        
        # UI state tracking
        self.current_detection = {}
        self.image_size = None
        
        # Measurement UI data
        self.frozen_frame = None
        self.frozen_frame_raw = None  # Raw (distorted) frame for accurate measurement math
        self.frozen_frame_is_raw = False  # Track if displayed frame is raw (True) or undistorted (False)
        self.click_points = []
        self.current_measure_frame = None
        
        # Calibration data (loaded from files)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None
        
        # Video thread
        self.video_thread = None
        
        self.init_ui()
        self.load_saved_calibration()

    def resource_path(self, path):
        """Get the absolute path to resource, works in PyInstaller exe"""
        if hasattr(sys, "_MEIPASS"):
            return os.path.join(sys._MEIPASS, path)
        return os.path.join(os.path.abspath("."), path)

    def load_runtime_settings(self):
        """Load persisted runtime settings from JSON cache file."""
        defaults = get_default_runtime_settings()

        if not os.path.exists(RUNTIME_SETTINGS_FILE):
            return defaults

        try:
            with open(RUNTIME_SETTINGS_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
            is_valid, _, normalized = validate_runtime_settings(data)
            if is_valid:
                return normalized
        except Exception:
            pass

        return defaults

    def save_runtime_settings(self):
        """Persist runtime settings to JSON cache file."""
        try:
            with open(RUNTIME_SETTINGS_FILE, "w", encoding="utf-8") as file:
                json.dump(self.runtime_settings, file, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Settings Warning", f"Failed to save settings cache: {str(e)}")

    def open_camera_parameters_popup(self):
        """Show popup and update runtime settings if user confirms."""
        dialog = CameraParametersDialog(self.runtime_settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False

        is_valid, message, normalized = validate_runtime_settings(dialog.get_settings())
        if not is_valid:
            QMessageBox.warning(self, "Invalid Parameters", message)
            return False

        self.runtime_settings = normalized
        self.save_runtime_settings()
        return True

    def apply_runtime_detectors(self):
        """Recreate board and detectors from current runtime settings."""
        settings = self.runtime_settings
        self.charuco_detector, self.aruco_detector, self.board, self.aruco_dict = create_detectors(
            settings["dict_type"],
            settings["squares_x"],
            settings["squares_y"],
            settings["square_length"],
            settings["marker_length"],
        )

    def create_video_thread(self):
        """Create a configured video thread from current runtime settings."""
        settings = self.runtime_settings
        return VideoThread(
            self.camera_source,
            self.charuco_detector,
            self.board,
            invert_colors=settings["invert_colors"],
            frame_width=settings["frame_width"],
            frame_height=settings["frame_height"],
        )

    def stop_active_camera(self):
        """Stop any active camera thread safely"""
        if self.video_thread and self.video_thread.isRunning():
            try:
                self.video_thread.change_pixmap.disconnect()
            except TypeError:
                pass
            try:
                self.video_thread.detection_info.disconnect()
            except TypeError:
                pass
            try:
                self.video_thread.camera_lost.disconnect()
            except TypeError:
                pass
            try:
                self.video_thread.finished.disconnect()
            except TypeError:
                pass

            self.video_thread.stop()
            self.video_thread = None

        # Re-enable camera selector
        if hasattr(self, 'camera_selector'):
            self.camera_selector.setEnabled(True)
            self.refresh_cam_btn.setEnabled(True)
            self.camera_selector.setToolTip("")
            self.refresh_cam_btn.setToolTip("")

    def _reset_tab_buttons_for_stopped_camera(self):
        """Reset camera-related controls based on current tab after stream stops."""
        current_tab = self.tabs.currentIndex() if hasattr(self, 'tabs') else 0

        if current_tab == 0:  # Intrinsics
            self.calib_start_btn.setEnabled(True)
            self.calib_capture_btn.setEnabled(False)
            self.calib_stop_btn.setEnabled(False)
        elif current_tab == 1:  # Extrinsics
            self.extrin_start_btn.setEnabled(True)
            self.extrin_capture_btn.setEnabled(False)
            self.extrin_stop_btn.setEnabled(False)
        elif current_tab == 2:  # Measurement
            self.measure_start_btn.setEnabled(True)
            self.measure_freeze_btn.setEnabled(False)
            self.measure_stop_btn.setEnabled(False)

    def on_video_thread_finished(self):
        """Handle camera thread completion (normal stop or unexpected end)."""
        self.video_thread = None
        if hasattr(self, 'camera_selector'):
            self.camera_selector.setEnabled(True)
            self.refresh_cam_btn.setEnabled(True)
            self.camera_selector.setToolTip("")
            self.refresh_cam_btn.setToolTip("")
        self._reset_tab_buttons_for_stopped_camera()

    def on_camera_lost(self, message):
        """Handle sudden camera disconnect/loss from worker thread."""
        self.on_video_thread_finished()
        self.statusBar().showMessage(message, 5000)
        QMessageBox.warning(self, "Camera Disconnected", message)

    def on_frame_timing_update(self, timing_data):
        """Handle frame timing diagnostics from video thread.
        
        Args:
            timing_data (dict): {'avg_process_time_ms': float, 'dropped_frames': int, 'frame_width': int, 'frame_height': int}
        """
        avg_time = timing_data.get('avg_process_time_ms', 0)
        dropped = timing_data.get('dropped_frames', 0)
        width = timing_data.get('frame_width', 0)
        height = timing_data.get('frame_height', 0)
        
        # Check if processing is slow (> 40ms for 30fps target)
        # At 4K+ resolutions, frame processing can exceed hardware capabilities
        if avg_time > 40:
            warning_msg = f"⚠ High frame processing time: {avg_time:.1f}ms (resolution: {width}×{height})"
            if dropped > 0:
                warning_msg += f" | Dropped {dropped} frames"
            
            # Only warn if resolution is high (likely to be problematic)
            if width >= 1920 or height >= 1440:
                self.statusBar().showMessage(warning_msg, 2000)

    def init_detectors(self):
        """Initialize ArUco and ChArUco detectors"""
        try:
            self.apply_runtime_detectors()
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", 
                               f"Failed to initialize detectors: {str(e)}")
            sys.exit(1)
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("ChArUco Calibration & Measurement Tool")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_calibration_tab(), "1. Intrinsics Calibration")
        self.tabs.addTab(self.create_extrinsics_tab(), "2. Extrinsics Capture")
        self.tabs.addTab(self.create_measurement_tab(), "3. Measurement")
        main_layout.addWidget(self.tabs)

        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # Status bar
        self.statusBar().showMessage("Ready")

        # Camera selector in status bar
        self.statusBar().addPermanentWidget(QLabel("Camera: "))
        self.camera_selector = QComboBox()
        self.camera_selector.setMinimumWidth(200)
        self.refresh_cam_btn = QPushButton("Refresh")
        self.refresh_cam_btn.setMaximumWidth(60)
        self.statusBar().addPermanentWidget(self.camera_selector)
        self.statusBar().addPermanentWidget(self.refresh_cam_btn)

        # Populate initial camera list
        self.refresh_camera_list()

        # Connect signals
        self.camera_selector.currentIndexChanged.connect(self.on_camera_selected)
        self.refresh_cam_btn.clicked.connect(self.refresh_camera_list)

    def refresh_camera_list(self):
        """Refresh the camera selector dropdown with available cameras"""
        self.camera_selector.blockSignals(True)  # Prevent triggering during update
        self.camera_selector.clear()

        try:
            cameras = get_available_cameras_with_names()

            if not cameras:
                self.camera_selector.addItem("No cameras detected", None)
                self.statusBar().showMessage("No cameras found", 5000)
            else:
                for cam in cameras:
                    self.camera_selector.addItem(cam['display_name'], cam['source'])

                # Try to select the current camera source
                current_source = self.camera_source
                selected = False
                for i in range(self.camera_selector.count()):
                    if self.camera_selector.itemData(i) == current_source:
                        self.camera_selector.setCurrentIndex(i)
                        selected = True
                        break

                # If current camera not found, select first available
                if not selected and self.camera_selector.count() > 0:
                    self.camera_selector.setCurrentIndex(0)
                    self.camera_source = self.camera_selector.itemData(0)

                self.statusBar().showMessage(f"Found {len(cameras)} camera(s)")
        except Exception as e:
            self.statusBar().showMessage(f"Error detecting cameras: {str(e)}")
        finally:
            self.camera_selector.blockSignals(False)

    def on_camera_selected(self, index):
        """Handle camera selection change"""
        cam_id = self.camera_selector.itemData(index)
        if cam_id is not None:
            # Validate camera can be opened
            try:
                test_cap = cv2.VideoCapture(cam_id, get_camera_backend())
                if not test_cap.isOpened():
                    raise RuntimeError("Cannot open camera")
                test_cap.release()
            except Exception as e:
                QMessageBox.warning(self, "Camera Unavailable",
                                  f"Cannot access camera {cam_id}. It may be in use by another application.")
                return

            self.camera_source = cam_id
            self.statusBar().showMessage(f"Camera selected: {cam_id}")
            # Stop any active camera when changing selection
            self.stop_active_camera()

    def on_tab_changed(self, index):
        """Stop camera when switching tabs"""
        self.stop_active_camera()

        # Reset UI state for all tabs
        self.calib_start_btn.setEnabled(True)
        self.calib_capture_btn.setEnabled(False)
        self.calib_stop_btn.setEnabled(False)

        self.extrin_start_btn.setEnabled(True)
        self.extrin_capture_btn.setEnabled(False)
        self.extrin_stop_btn.setEnabled(False)

        self.measure_start_btn.setEnabled(True)
        self.measure_freeze_btn.setEnabled(False)
        self.measure_stop_btn.setEnabled(False)

        self.statusBar().showMessage("Camera stopped (tab changed)")
        
    def create_calibration_tab(self):
        """Create the calibration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QGroupBox("Instructions")
        inst_layout = QVBoxLayout()
        inst_text = QLabel(
            "1. Click 'Start Camera' to begin live preview\n"
            "2. Show ChArUco board to camera from various angles\n"
            "3. Click 'Capture Frame' when board is clearly visible\n"
            "4. Collect 20-25 frames with different orientations\n"
            "5. Click 'Calibrate' to compute camera intrinsics"
        )
        inst_text.setWordWrap(True)
        inst_layout.addWidget(inst_text)
        instructions.setLayout(inst_layout)
        layout.addWidget(instructions)
        
        # Video display
        self.calib_video_label = QLabel()
        self.calib_video_label.setMinimumSize(640, 480)
        self.calib_video_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.calib_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.calib_video_label.setText("Camera feed will appear here")
        layout.addWidget(self.calib_video_label)
        
        # Info panel
        info_layout = QHBoxLayout()
        self.calib_corners_label = QLabel("Corners Detected: 0")
        self.calib_frames_label = QLabel("Frames Captured: 0")
        self.calib_progress = QProgressBar()
        self.calib_progress.setMaximum(25)  # Target 25 frames for calibration
        info_layout.addWidget(self.calib_corners_label)
        info_layout.addWidget(self.calib_frames_label)
        info_layout.addWidget(self.calib_progress)
        layout.addLayout(info_layout)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.calib_start_btn = QPushButton("Start Camera")
        self.calib_start_btn.clicked.connect(self.start_calibration_camera)
        self.calib_capture_btn = QPushButton("Capture Frame")
        self.calib_capture_btn.clicked.connect(self.capture_calibration_frame)
        self.calib_capture_btn.setEnabled(False)
        self.calib_calibrate_btn = QPushButton("Calibrate")
        self.calib_calibrate_btn.clicked.connect(self.run_calibration)
        self.calib_calibrate_btn.setEnabled(False)
        self.calib_stop_btn = QPushButton("Stop Camera")
        self.calib_stop_btn.clicked.connect(self.stop_calibration_camera)
        self.calib_stop_btn.setEnabled(False)
        
        btn_layout.addWidget(self.calib_start_btn)
        btn_layout.addWidget(self.calib_capture_btn)
        btn_layout.addWidget(self.calib_calibrate_btn)
        btn_layout.addWidget(self.calib_stop_btn)
        layout.addLayout(btn_layout)
        
        # Log
        self.calib_log = QTextEdit()
        self.calib_log.setMaximumHeight(100)
        self.calib_log.setReadOnly(True)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.calib_log)
        
        return tab
    
    def create_extrinsics_tab(self):
        """Create the extrinsics capture tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QGroupBox("Instructions")
        inst_layout = QVBoxLayout()
        inst_text = QLabel(
            "1. Ensure intrinsics calibration is completed first\n"
            "2. Place ChArUco board on the measurement plane\n"
            "3. Click 'Start Camera' and position board clearly in view\n"
            "4. Click 'Capture & Compute Extrinsics' to save board pose\n"
            "⚠️ Camera must remain FIXED after this step!"
        )
        inst_text.setWordWrap(True)
        inst_layout.addWidget(inst_text)
        instructions.setLayout(inst_layout)
        layout.addWidget(instructions)
        
        # Video display
        self.extrin_video_label = QLabel()
        self.extrin_video_label.setMinimumSize(640, 480)
        self.extrin_video_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.extrin_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.extrin_video_label.setText("Camera feed will appear here")
        layout.addWidget(self.extrin_video_label)
        
        # Info
        self.extrin_status_label = QLabel("Status: Not calibrated")
        layout.addWidget(self.extrin_status_label)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.extrin_start_btn = QPushButton("Start Camera")
        self.extrin_start_btn.clicked.connect(self.start_extrinsics_camera)
        self.extrin_capture_btn = QPushButton("Capture & Compute Extrinsics")
        self.extrin_capture_btn.clicked.connect(self.capture_extrinsics)
        self.extrin_capture_btn.setEnabled(False)
        self.extrin_stop_btn = QPushButton("Stop Camera")
        self.extrin_stop_btn.clicked.connect(self.stop_extrinsics_camera)
        self.extrin_stop_btn.setEnabled(False)
        
        btn_layout.addWidget(self.extrin_start_btn)
        btn_layout.addWidget(self.extrin_capture_btn)
        btn_layout.addWidget(self.extrin_stop_btn)
        layout.addLayout(btn_layout)
        
        # Log
        self.extrin_log = QTextEdit()
        self.extrin_log.setMaximumHeight(100)
        self.extrin_log.setReadOnly(True)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.extrin_log)
        
        return tab
    
    def create_measurement_tab(self):
        """Create the measurement tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QGroupBox("Instructions")
        inst_layout = QVBoxLayout()
        inst_text = QLabel(
            "1. Ensure both intrinsics and extrinsics are calibrated\n"
            "2. Click 'Start Measurement' to begin\n"
            "3. Click 'Freeze Frame' to capture current view\n"
            "4. Click two points on the frozen frame to measure distance\n"
            "5. Results will appear in the log below"
        )
        inst_text.setWordWrap(True)
        inst_layout.addWidget(inst_text)
        instructions.setLayout(inst_layout)
        layout.addWidget(instructions)
        
        # Video display
        self.measure_video_label = QLabel()
        self.measure_video_label.setMinimumSize(640, 480)
        self.measure_video_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.measure_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.measure_video_label.setText("Camera feed will appear here")
        self.measure_video_label.mousePressEvent = self.measurement_click
        layout.addWidget(self.measure_video_label)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.measure_start_btn = QPushButton("Start Measurement")
        self.measure_start_btn.clicked.connect(self.start_measurement)
        self.measure_freeze_btn = QPushButton("Freeze Frame")
        self.measure_freeze_btn.clicked.connect(self.freeze_frame)
        self.measure_freeze_btn.setEnabled(False)
        self.measure_stop_btn = QPushButton("Stop Measurement")
        self.measure_stop_btn.clicked.connect(self.stop_measurement)
        self.measure_stop_btn.setEnabled(False)
        
        btn_layout.addWidget(self.measure_start_btn)
        btn_layout.addWidget(self.measure_freeze_btn)
        btn_layout.addWidget(self.measure_stop_btn)

        # Undistortion toggle
        self.undistort_checkbox = QCheckBox("Apply undistortion")
        self.undistort_checkbox.setChecked(False)  # Default OFF for low-distortion cameras
        self.undistort_checkbox.setToolTip("Toggle to apply lens distortion correction.\n"
                                          "Uncheck for cameras with minimal distortion.")
        btn_layout.addWidget(self.undistort_checkbox)

        layout.addLayout(btn_layout)
        
        # Log
        self.measure_log = QTextEdit()
        self.measure_log.setMaximumHeight(150)
        self.measure_log.setReadOnly(True)
        layout.addWidget(QLabel("Measurements:"))
        layout.addWidget(self.measure_log)
        
        return tab
    
    # ==================== CALIBRATION FUNCTIONS ====================
    
    def start_calibration_camera(self):
        """Start camera for calibration"""
        try:
            # Check if a valid camera is selected
            if self.camera_selector.currentData() is None:
                QMessageBox.warning(self, "No Camera",
                                  "No cameras detected. Please connect a camera and click Refresh.")
                return

            if self.video_thread and self.video_thread.isRunning():
                self.stop_calibration_camera()

            if not self.open_camera_parameters_popup():
                self.log_message(self.calib_log, "Camera start cancelled")
                return

            self.apply_runtime_detectors()

            self.video_thread = self.create_video_thread()
            self.video_thread.change_pixmap.connect(self.update_calib_image)
            self.video_thread.detection_info.connect(self.update_calib_detection)
            self.video_thread.camera_lost.connect(self.on_camera_lost)
            self.video_thread.frame_timing.connect(self.on_frame_timing_update)
            self.video_thread.finished.connect(self.on_video_thread_finished)
            self.video_thread.start()

            # Disable camera selector while camera is running
            self.camera_selector.setEnabled(False)
            self.refresh_cam_btn.setEnabled(False)
            self.camera_selector.setToolTip("Stop camera to change selection")
            self.refresh_cam_btn.setToolTip("Stop camera to refresh")

            self.calib_start_btn.setEnabled(False)
            self.calib_capture_btn.setEnabled(True)
            self.calib_stop_btn.setEnabled(True)
            settings = self.runtime_settings
            self.log_message(
                self.calib_log,
                f"Camera started (source: {self.camera_source}, {settings['frame_width']}x{settings['frame_height']}, {settings['dict_type']})",
            )

        except Exception as e:
            QMessageBox.critical(self, "Camera Error", f"Failed to start camera: {str(e)}")
    
    def update_calib_image(self, image):
        """Update calibration video display"""
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(self.calib_video_label.size(), 
                              Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        self.calib_video_label.setPixmap(scaled)
    
    def update_calib_detection(self, info):
        """Update detection information"""
        self.current_detection = info
        self.calib_corners_label.setText(f"Corners Detected: {info['corners_detected']}")
        
        if self.image_size is None and 'frame' in info:
            frame = info['frame']
            self.image_size = (frame.shape[1], frame.shape[0])
    
    def capture_calibration_frame(self):
        """Capture current frame for calibration"""
        try:
            if self.current_detection.get('corners_detected', 0) < MIN_CHARUCO_CORNERS:
                QMessageBox.warning(self, "Insufficient Corners", 
                                  f"Need at least {MIN_CHARUCO_CORNERS} corners. Current: {self.current_detection.get('corners_detected', 0)}")
                return
            
            # Use intrinsic calibration module
            frame_count = self.intrinsic.add_frame(
                self.current_detection['charuco_corners'],
                self.current_detection['charuco_ids'],
                self.image_size
            )
            
            self.calib_frames_label.setText(f"Frames Captured: {frame_count}")
            self.calib_progress.setValue(frame_count)
            self.log_message(self.calib_log, 
                           f"Captured frame {frame_count} with {self.current_detection['corners_detected']} corners")
            
            if frame_count >= 8:
                self.calib_calibrate_btn.setEnabled(True)
                
        except Exception as e:
            QMessageBox.critical(self, "Capture Error", f"Failed to capture frame: {str(e)}")
    
    def run_calibration(self):
        """Run camera calibration"""
        try:
            if self.intrinsic.get_frame_count() < 8:
                QMessageBox.warning(self, "Insufficient Frames", 
                                  "Collect at least 8 frames for calibration")
                return
            
            self.calib_calibrate_btn.setEnabled(False)
            self.log_message(self.calib_log, "Running calibration... Please wait.")
            
            # Get frame dimensions from current runtime settings for adaptive RMS threshold
            frame_width = self.runtime_settings.get('frame_width', 1280)
            frame_height = self.runtime_settings.get('frame_height', 960)
            
            # Run calibration in background thread using module
            self.calib_worker = self.intrinsic.run_calibration(self.board, frame_width, frame_height)
            self.calib_worker.progress.connect(lambda msg: self.log_message(self.calib_log, msg))
            self.calib_worker.finished.connect(self.calibration_finished)
            self.calib_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Calibration Error", f"Calibration failed: {str(e)}")
            self.calib_calibrate_btn.setEnabled(True)
    
    def calibration_finished(self, success, message, data):
        """Handle calibration completion"""
        if success:
            try:
                # Save using intrinsic module
                filepath = self.intrinsic.save_calibration(data)
                
                self.log_message(self.calib_log, f"✓ {message}")
                self.log_message(self.calib_log, f"✓ Saved to {filepath}")
                
                QMessageBox.information(self, "Success", 
                                      f"Calibration completed!\n{message}\n\nProceed to Extrinsics tab.")
                
                self.load_saved_calibration()
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save calibration: {str(e)}")
        else:
            self.log_message(self.calib_log, f"✗ {message}")
            QMessageBox.warning(self, "Calibration Failed", message)
            self.calib_calibrate_btn.setEnabled(True)
    
    def stop_calibration_camera(self):
        """Stop calibration camera"""
        self.stop_active_camera()

        self.calib_start_btn.setEnabled(True)
        self.calib_capture_btn.setEnabled(False)
        self.calib_stop_btn.setEnabled(False)
        self.log_message(self.calib_log, "Camera stopped")
    
    # ==================== EXTRINSICS FUNCTIONS ====================
    
    def start_extrinsics_camera(self):
        """Start camera for extrinsics"""
        try:
            # Check if a valid camera is selected
            if self.camera_selector.currentData() is None:
                QMessageBox.warning(self, "No Camera",
                                  "No cameras detected. Please connect a camera and click Refresh.")
                return

            if not os.path.exists(CALIB_FILE):
                QMessageBox.warning(self, "No Calibration",
                                  "Complete intrinsics calibration first!")
                return

            if self.video_thread and self.video_thread.isRunning():
                self.stop_extrinsics_camera()

            if not self.open_camera_parameters_popup():
                self.log_message(self.extrin_log, "Camera start cancelled")
                return

            self.apply_runtime_detectors()

            # Load intrinsics for extrinsic module
            self.camera_matrix, self.dist_coeffs = self.extrinsic.load_intrinsics()
            self.extrinsic.set_intrinsics(self.camera_matrix, self.dist_coeffs)

            self.video_thread = self.create_video_thread()
            self.video_thread.change_pixmap.connect(self.update_extrin_image)
            self.video_thread.detection_info.connect(self.update_extrin_detection)
            self.video_thread.camera_lost.connect(self.on_camera_lost)
            self.video_thread.frame_timing.connect(self.on_frame_timing_update)
            self.video_thread.finished.connect(self.on_video_thread_finished)
            self.video_thread.start()

            # Disable camera selector while camera is running
            self.camera_selector.setEnabled(False)
            self.refresh_cam_btn.setEnabled(False)
            self.camera_selector.setToolTip("Stop camera to change selection")
            self.refresh_cam_btn.setToolTip("Stop camera to refresh")

            self.extrin_start_btn.setEnabled(False)
            self.extrin_capture_btn.setEnabled(True)
            self.extrin_stop_btn.setEnabled(True)
            settings = self.runtime_settings
            self.log_message(
                self.extrin_log,
                f"Camera started - show ChArUco board ({settings['frame_width']}x{settings['frame_height']}, {settings['dict_type']})",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start: {str(e)}")
    
    def update_extrin_image(self, image):
        """Update extrinsics video display"""
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(self.extrin_video_label.size(),
                              Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        self.extrin_video_label.setPixmap(scaled)
    
    def update_extrin_detection(self, info):
        """Update extrinsics detection info"""
        self.current_detection = info
        self.extrin_status_label.setText(f"Corners Detected: {info['corners_detected']}")
    
    def capture_extrinsics(self):
        """Capture and compute extrinsics"""
        try:
            if self.current_detection.get('corners_detected', 0) < MIN_CHARUCO_CORNERS:
                QMessageBox.warning(self, "Insufficient Corners",
                                  f"Need at least {MIN_CHARUCO_CORNERS} ChArUco corners. Current: {self.current_detection.get('corners_detected', 0)}")
                return
            
            # Use extrinsic module to capture pose
            success, message, rvec, tvec = self.extrinsic.capture_pose(
                self.current_detection['charuco_corners'],
                self.current_detection['charuco_ids'],
                self.board
            )
            
            if success:
                # Save extrinsics
                filepath = self.extrinsic.save_extrinsics(rvec, tvec)
                
                self.log_message(self.extrin_log, f"✓ {message}")
                self.log_message(self.extrin_log, f"✓ Saved to {filepath}")
                self.extrin_status_label.setText("Status: ✓ Extrinsics captured")
                
                QMessageBox.information(self, "Success",
                                      f"{message}\n\nProceed to Measurement tab.")
                
                self.load_saved_calibration()
            else:
                self.log_message(self.extrin_log, f"✗ {message}")
                QMessageBox.warning(self, "Capture Failed", message)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to capture extrinsics: {str(e)}")
    
    def stop_extrinsics_camera(self):
        """Stop extrinsics camera"""
        self.stop_active_camera()
        
        self.extrin_start_btn.setEnabled(True)
        self.extrin_capture_btn.setEnabled(False)
        self.extrin_stop_btn.setEnabled(False)
        self.log_message(self.extrin_log, "Camera stopped")
    
    # ==================== MEASUREMENT FUNCTIONS ====================
    
    def start_measurement(self):
        """Start measurement mode"""
        try:
            # Check if a valid camera is selected
            if self.camera_selector.currentData() is None:
                QMessageBox.warning(self, "No Camera",
                                  "No cameras detected. Please connect a camera and click Refresh.")
                return

            if not self.load_saved_calibration():
                QMessageBox.warning(self, "Not Calibrated",
                                  "Complete both intrinsics and extrinsics calibration first!")
                return

            if self.video_thread and self.video_thread.isRunning():
                self.stop_measurement()

            if not self.open_camera_parameters_popup():
                self.log_message(self.measure_log, "Measurement start cancelled")
                return

            self.apply_runtime_detectors()

            # Set calibration data in measurement module
            self.measurement.set_calibration(
                self.camera_matrix, self.dist_coeffs,
                self.rvec, self.tvec
            )

            self.video_thread = self.create_video_thread()
            self.video_thread.change_pixmap.connect(self.update_measure_display_from_signal)
            self.video_thread.detection_info.connect(self.store_measurement_frame)
            self.video_thread.camera_lost.connect(self.on_camera_lost)
            self.video_thread.frame_timing.connect(self.on_frame_timing_update)
            self.video_thread.finished.connect(self.on_video_thread_finished)
            self.video_thread.start()

            # Disable camera selector while camera is running
            self.camera_selector.setEnabled(False)
            self.refresh_cam_btn.setEnabled(False)
            self.camera_selector.setToolTip("Stop camera to change selection")
            self.refresh_cam_btn.setToolTip("Stop camera to refresh")

            self.measure_start_btn.setEnabled(False)
            self.measure_freeze_btn.setEnabled(True)
            self.measure_stop_btn.setEnabled(True)
            self.frozen_frame = None
            self.click_points = []
            self.current_measure_frame = None

            settings = self.runtime_settings
            self.log_message(
                self.measure_log,
                f"Measurement mode started ({settings['frame_width']}x{settings['frame_height']}, {settings['dict_type']})",
            )

        except Exception as e:
            QMessageBox.critical(self, "Measurement Error", f"Failed to start: {str(e)}")
    
    def store_measurement_frame(self, info):
        """Store the current raw frame and update display with undistorted version"""
        if 'frame' in info:
            self.current_measure_frame = info['frame']

            # Update display with undistorted frame (if not frozen)
            if self.frozen_frame is None:
                self.update_measure_display()
    
    def update_measure_display_from_signal(self, image):
        """Update measurement video display from change_pixmap signal - displays distorted feed as fallback"""
        # The actual undistorted display is handled in store_measurement_frame
        # This is a fallback in case detection_info signal is delayed
        if self.frozen_frame is not None or self.current_measure_frame is None:
            return
        # Only update if we haven't already updated via store_measurement_frame
        # (store_measurement_frame runs first since detection_info is emitted before change_pixmap)
        pass  # Display is handled by store_measurement_frame -> update_measure_display

    def update_measure_display(self):
        """Update measurement video display - respects undistortion toggle"""
        # Skip if frame is frozen (user clicked Freeze Frame)
        if self.frozen_frame is not None:
            return

        if self.current_measure_frame is None:
            return

        # Check if undistortion should be applied
        apply_undistortion = self.undistort_checkbox.isChecked()

        if apply_undistortion and self.camera_matrix is not None and self.dist_coeffs is not None:
            # Undistort using measurement module
            display_frame = self.measurement.undistort_frame(self.current_measure_frame)
            overlay_text = "Undistorted Feed"
        else:
            # Show raw frame
            display_frame = self.current_measure_frame.copy()
            overlay_text = "Raw Feed"

        # Add text overlay
        cv2.putText(display_frame, overlay_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Convert to Qt format
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)

        scaled = pixmap.scaled(self.measure_video_label.size(),
                              Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        self.measure_video_label.setPixmap(scaled)
    
    def freeze_frame(self):
        """Freeze current frame for measurement"""
        if self.video_thread and self.video_thread.isRunning():
            if self.current_measure_frame is not None:
                # Store RAW frame - needed for accurate measurement math
                self.frozen_frame_raw = self.current_measure_frame.copy()

                # Check if undistortion should be applied for display
                apply_undistortion = self.undistort_checkbox.isChecked()
                self.frozen_frame_is_raw = not apply_undistortion

                if apply_undistortion and self.camera_matrix is not None and self.dist_coeffs is not None:
                    # Create undistorted version for display
                    display_frame = self.measurement.undistort_frame(self.current_measure_frame)
                else:
                    # Show raw frame
                    display_frame = self.current_measure_frame.copy()

                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.frozen_frame = QPixmap.fromImage(qt_image)
            else:
                self.frozen_frame = self.measure_video_label.pixmap()
                self.frozen_frame_raw = None
                self.frozen_frame_is_raw = False

            self.video_thread.stop()
            self.video_thread = None

            self.click_points = []
            self.measurement.reset_points()
            self.measure_freeze_btn.setEnabled(False)
            self.log_message(self.measure_log, "Frame frozen - click any two points to measure")
    
    def measurement_click(self, event):
        """Handle mouse clicks on measurement view"""
        if self.frozen_frame is None:
            return
        
        if len(self.click_points) >= 2:
            # Start a new measurement pair on next click without requiring reset/restart.
            self.click_points = []
            self.measurement.reset_points()
            scaled = self.frozen_frame.scaled(self.measure_video_label.size(),
                                             Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation)
            self.measure_video_label.setPixmap(scaled)
        
        # Get click position relative to actual image
        label_size = self.measure_video_label.size()
        pixmap_size = self.frozen_frame.size()
        
        # Calculate scaling and offset
        scale_w = pixmap_size.width() / label_size.width()
        scale_h = pixmap_size.height() / label_size.height()
        scale = max(scale_w, scale_h)
        
        offset_x = (label_size.width() - pixmap_size.width() / scale) / 2
        offset_y = (label_size.height() - pixmap_size.height() / scale) / 2
        
        # Convert click to image coordinates
        img_x = int((event.pos().x() - offset_x) * scale)
        img_y = int((event.pos().y() - offset_y) * scale)
        
        # Store point in both UI and measurement module
        self.click_points.append((img_x, img_y))
        self.measurement.add_click_point((img_x, img_y))
        
        # Draw points on frozen frame
        self.draw_measurement_points()
        
        # If two points, compute distance
        if len(self.click_points) == 2:
            self.compute_distance()
    
    def draw_measurement_points(self):
        """Draw clicked points on frozen frame"""
        if self.frozen_frame is None:
            return
        
        # Convert pixmap to image for drawing
        image = self.frozen_frame.toImage()
        
        # Create a copy for drawing
        temp_pixmap = QPixmap.fromImage(image)
        painter = temp_pixmap.toImage()
        
        # Convert to cv2 format for easier drawing
        width = painter.width()
        height = painter.height()
        ptr = painter.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        
        # Draw points and line
        for i, (x, y) in enumerate(self.click_points):
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(frame, str(i+1), (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(self.click_points) == 2:
            cv2.line(frame, self.click_points[0], self.click_points[1], (255, 0, 0), 2)
        
        # Convert back to QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        scaled = pixmap.scaled(self.measure_video_label.size(),
                              Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        self.measure_video_label.setPixmap(scaled)
    
    def compute_distance(self):
        """Compute distance between two clicked points"""
        try:
            # Use measurement module to compute distance
            success, message, dist_m, dist_cm, p1, p2 = self.measurement.compute_distance()
            
            if not success:
                self.log_message(self.measure_log, f"✗ {message}")
                QMessageBox.warning(self, "Measurement Error", message)
                return
            
            dist_mm = dist_m * 1000
            self.log_message(self.measure_log, 
                           f"Distance: {dist_m:.6f} m  =  {dist_cm:.2f} cm  =  {dist_mm:.1f} mm")
            
            # Draw distance on image
            frame = self.frozen_frame.toImage()
            width = frame.width()
            height = frame.height()
            ptr = frame.bits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            
            # Draw line and text
            cv2.line(img, self.click_points[0], self.click_points[1], (255, 0, 0), 3)
            for i, (x, y) in enumerate(self.click_points):
                cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(img, str(i+1), (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            mid_x = (self.click_points[0][0] + self.click_points[1][0]) // 2
            mid_y = (self.click_points[0][1] + self.click_points[1][1]) // 2
            cv2.putText(img, f"{dist_cm:.2f} cm", (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            # Update display
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled = pixmap.scaled(self.measure_video_label.size(),
                                  Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
            self.measure_video_label.setPixmap(scaled)
            
        except Exception as e:
            self.log_message(self.measure_log, f"✗ Error computing distance: {str(e)}")
            QMessageBox.critical(self, "Measurement Error", f"Failed to compute distance: {str(e)}")
    
    def stop_measurement(self):
        """Stop measurement mode"""
        try:
            self.stop_active_camera()
        except Exception:
            self.video_thread = None
            self.log_message(self.measure_log, "camera stopping failed")
        
        self.measure_start_btn.setEnabled(True)
        self.measure_freeze_btn.setEnabled(False)
        self.measure_stop_btn.setEnabled(False)
        self.frozen_frame = None
        self.click_points = []
        self.current_measure_frame = None
        self.measurement.reset_points()
        self.log_message(self.measure_log, "Measurement stopped")
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def load_saved_calibration(self):
        """Load saved calibration data"""
        try:
            if os.path.exists(CALIB_FILE):
                self.camera_matrix, self.dist_coeffs = IntrinsicCalibration.load_calibration()
                self.statusBar().showMessage("✓ Intrinsics loaded")
            else:
                self.statusBar().showMessage("⚠ No intrinsics calibration found")
                return False
            
            if os.path.exists(EXTRINSICS_FILE):
                self.rvec, self.tvec = ExtrinsicCalibration.load_extrinsics()
                self.statusBar().showMessage("✓ Intrinsics and Extrinsics loaded")
                return True
            else:
                self.statusBar().showMessage("⚠ Extrinsics not found - complete extrinsics calibration")
                return False
                
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Error loading calibration: {str(e)}")
            return False
    
    def log_message(self, log_widget, message):
        """Add message to log widget"""
        log_widget.append(message)
        log_widget.verticalScrollBar().setValue(log_widget.verticalScrollBar().maximum())
    
    def closeEvent(self, event):
        """Handle window close"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application metadata
    app.setApplicationName("ChArUco Calibration Tool")
    app.setOrganizationName("Camera Calibration")
    
    window = ChArUcoCalibrationGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
