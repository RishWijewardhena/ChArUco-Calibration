# ChArUco Calibration Application - Modular Architecture

## Overview

This application has been refactored into a modular architecture with OS-aware camera handling. The original monolithic 1029-line file has been split into focused modules for better maintainability and cross-platform support.

## Architecture

### Module Structure

```
calibration App/
├── config.py                      # Configuration constants and detector initialization
├── camera_utils.py                # OS-aware camera handling (Windows/Linux/macOS)
├── video_thread.py                # QThread for video capture and ChArUco detection
├── intrinsic_calibration.py       # Intrinsic calibration workflow
├── extrinsic_calibration.py       # Extrinsic calibration workflow
├── measurements.py                # 3D measurement calculations
├── Calibration_App_refactored.py  # Main GUI application (NEW)
├── Calibration_App.py             # Original monolithic file (BACKUP)
├── test_modules.py                # Module verification script
└── requirements.txt               # Python dependencies
```

## New Features

### 1. **OS-Aware Camera Handling**

The application now automatically detects your operating system and uses the appropriate camera source:

- **Windows**: Camera index `0` with DirectShow backend (CAP_DSHOW)
- **Linux**: Device path `/dev/video0` with V4L2 backend (CAP_V4L2)
- **macOS**: Camera index `0` with AVFoundation backend (CAP_AVFOUNDATION)

**Environment Override**: Set `CAMERA_SOURCE` environment variable to use a custom camera:
```bash
# Linux - use different video device
export CAMERA_SOURCE=/dev/video2
python Calibration_App_refactored.py

# Windows - use second webcam
set CAMERA_SOURCE=1
python Calibration_App_refactored.py
```

### 2. **Modular Design**

Each module has a clear responsibility:

#### **config.py**
- All ChArUco board parameters (dictionary type, dimensions, physical sizes)
- File paths for calibration data
- Detector initialization functions

#### **camera_utils.py**
- `get_camera_source()` - Returns OS-appropriate camera source
- `get_camera_backend()` - Returns OpenCV backend for the OS
- `open_camera()` - Opens camera with correct backend
- `get_available_cameras()` - Detects available camera devices

#### **video_thread.py**
- `VideoThread` class - QThread for non-blocking video capture
- Emits frames for display and detection info for processing

#### **intrinsic_calibration.py**
- `CalibrationWorker` - Background thread for calibration computation
- `IntrinsicCalibration` - Manages frame collection and calibration workflow
- Saves/loads camera matrix and distortion coefficients

#### **extrinsic_calibration.py**
- `ExtrinsicCalibration` - Manages board pose estimation
- Uses solvePnP to compute rotation and translation vectors
- Saves/loads extrinsic parameters

#### **measurements.py**
- `Measurement` - Handles 3D distance calculations
- Projects 2D image points to 3D measurement plane
- Applies lens distortion correction

#### **Calibration_App_refactored.py**
- Main GUI orchestrator
- Delegates business logic to workflow modules
- Handles Qt widgets, signals, and user interactions

## Installation

### Prerequisites

```bash
# Ensure Python 3.8+ is installed
python --version

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- PyQt6 (GUI framework)
- opencv-contrib-python (Computer vision with ArUco support)
- numpy (Numerical computations)

## Usage

### Option 1: Run Refactored Version Directly

```bash
python Calibration_App_refactored.py
```

### Option 2: Replace Original File

```bash
# Backup original
mv Calibration_App.py Calibration_App_backup.py

# Use refactored version as main
mv Calibration_App_refactored.py Calibration_App.py

# Run as usual
python Calibration_App.py
```

### Option 3: Test Before Running

```bash
# Verify all modules work correctly
python test_modules.py

# If tests pass, run the application
python Calibration_App_refactored.py
```

## Workflow

### Step 1: Intrinsic Calibration
1. Click **Start Camera** in "Intrinsics Calibration" tab
2. Show ChArUco board from various angles and distances
3. Click **Capture Frame** when board is clearly visible (15-30 frames recommended)
4. Click **Calibrate** to compute camera matrix and distortion coefficients
5. Results saved to `camera_calibration.json`

### Step 2: Extrinsic Calibration
1. Place ChArUco board flat on measurement surface
2. Click **Start Camera** in "Extrinsics Capture" tab
3. Ensure board is fully visible and well-lit
4. Click **Capture & Compute Extrinsics**
5. **⚠️ Do not move camera after this step!**
6. Results saved to `camera_extrinsics.json`

### Step 3: Measurement
1. Click **Start Measurement** in "Measurement" tab
2. View undistorted live feed
3. Click **Freeze Frame** to capture current view
4. Click two points on frozen frame
5. Distance displayed in meters, centimeters, and millimeters

## Configuration

### Custom ChArUco Board

Edit `config.py` to match your printed board:

```python
# ChArUco Dictionary
DICT_TYPE = cv2.aruco.DICT_4X4_50

# Board layout
SQUARES_X = 5  # Horizontal squares
SQUARES_Y = 6  # Vertical squares

# Physical dimensions (in meters)
SQUARE_LENGTH = 0.010  # Side length of black/white squares
MARKER_LENGTH = 0.008  # Side length of ArUco markers

# Detection threshold
MIN_CHARUCO_CORNERS = 4  # Minimum corners for valid detection
```

### Custom Camera Source

```bash
# Temporary override (single session)
export CAMERA_SOURCE=1  # Use second camera

# Permanent override (add to ~/.bashrc or ~/.zshrc)
echo 'export CAMERA_SOURCE=/dev/video2' >> ~/.bashrc
```

## Calibration Files

### camera_calibration.json (Intrinsics)
```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [[k1, k2, p1, p2, k3]],
  "rms": 0.5432,
  "image_size": [1280, 720]
}
```

### camera_extrinsics.json (Board Pose)
```json
{
  "rvec": [rx, ry, rz],
  "tvec": [tx, ty, tz]
}
```

## Troubleshooting

### Camera Not Found

**Linux:**
```bash
# List available video devices
ls -l /dev/video*

# Check permissions
sudo usermod -aG video $USER
# Logout and login again

# Test camera with v4l-utils
v4l2-ctl --list-devices
```

**Windows:**
```bash
# Try different camera indices
set CAMERA_SOURCE=0
python Calibration_App_refactored.py

set CAMERA_SOURCE=1
python Calibration_App_refactored.py
```

**macOS:**
```bash
# Grant camera permissions in System Preferences
# Security & Privacy → Camera → Allow Python
```

### Import Errors

```bash
# Ensure all modules are in the same directory
ls -l *.py

# Verify Python can find modules
python -c "import config; print('OK')"
```

### Low Calibration Quality

- Collect 20-30 frames (not just minimum 8)
- Vary board angles: tilted, rotated, near, far
- Ensure good lighting and sharp focus
- RMS error should be < 1.0 (lower is better)

### Measurement Inaccuracy

- Use high-resolution printed board (laser printer recommended)
- Measure physical board dimensions precisely
- Ensure flat measurement surface
- Keep camera steady after extrinsic calibration
- Verify `SQUARE_LENGTH` and `MARKER_LENGTH` in config.py

## Creating Executable (Windows/Linux)

```bash
# Install PyInstaller
pip install pyinstaller

# Create standalone executable
pyinstaller --onefile --windowed \
    --name "ChArUco_Calibration" \
    --icon "app.ico" \
    --add-data "config.py:." \
    --add-data "camera_utils.py:." \
    --add-data "video_thread.py:." \
    --add-data "intrinsic_calibration.py:." \
    --add-data "extrinsic_calibration.py:." \
    --add-data "measurements.py:." \
    Calibration_App_refactored.py

# Executable will be in dist/ folder
```

## Migration from Original Version

The refactored version is **100% backward compatible**:

- Uses same JSON file formats
- No changes to calibration algorithms
- Existing calibration files work without modification

**To switch:**
1. Backup original: `cp Calibration_App.py Calibration_App_original.py`
2. Test modules: `python test_modules.py`
3. Test GUI: `python Calibration_App_refactored.py`
4. If satisfied, rename: `mv Calibration_App_refactored.py Calibration_App.py`

## Development

### Adding New Features

**Example: Add new calibration method**

1. Create function in appropriate module (e.g., `intrinsic_calibration.py`)
2. Add UI elements in `Calibration_App_refactored.py`
3. Connect signals to new functions
4. Update this README

### Code Style

- PEP 8 compliant
- Type hints in function signatures (recommended)
- Docstrings for all classes and public methods
- Module-level docstring explaining purpose

### Testing

```bash
# Test individual modules
python -c "from intrinsic_calibration import IntrinsicCalibration; print('OK')"

# Test camera detection
python -c "from camera_utils import get_camera_source; print(get_camera_source())"

# Full module test
python test_modules.py
```

## Performance Considerations

- Video capture runs in separate QThread (non-blocking UI)
- Calibration computation runs in background thread
- Undistortion applied on-the-fly for measurements
- Typical calibration time: 5-10 seconds for 20 frames

## Known Limitations

- Requires physical ChArUco board (cannot use screen)
- Camera must remain stationary after extrinsic calibration
- Measurements only accurate on calibration plane (flat surface)
- Minimum 4 visible corners for detection

## Support

### Reporting Issues

Include the following information:
1. Operating system (Windows/Linux/macOS)
2. Python version: `python --version`
3. OpenCV version: `python -c "import cv2; print(cv2.__version__)"`
4. Camera source being used
5. Error message or unexpected behavior
6. Output of `python test_modules.py`

## License

This application is provided as-is for educational and research purposes.

## Version History

### Version 3.0 (Current - Modular Architecture)
- ✨ Modular design with separate workflow modules
- ✨ OS-aware camera handling (Windows/Linux/macOS)
- ✨ Environment variable camera override
- ✨ Improved code organization and maintainability
- ✨ Comprehensive test suite

### Version 2.0 (Original Monolithic)
- Desktop GUI with PyQt6
- Three-tab workflow (Intrinsics/Extrinsics/Measurement)
- ChArUco board calibration
- 3D distance measurement

---

**Built with Python, OpenCV, and PyQt6**
