"""
Configuration Module for ChArUco Calibration Application

Contains all configuration constants and detector initialization functions.
"""

import cv2

# ==================== FILE PATHS ====================
CALIB_FILE = "camera_calibration.json"
EXTRINSICS_FILE = "camera_extrinsics.json"
RUNTIME_SETTINGS_FILE = "runtime_settings.json"

# ==================== CHARUCO BOARD PARAMETERS ====================
DICT_TYPE = cv2.aruco.DICT_4X4_50
SQUARES_X = 6  # number of squares in X direction
SQUARES_Y = 5  # number of squares in Y direction
SQUARE_LENGTH = 0.010  # meters (adjust as needed)
MARKER_LENGTH = 0.008   # meters (adjust as needed)
MIN_CHARUCO_CORNERS = 4
FRAME_WIDTH = 1280
FRAME_HEIGHT = 960


# color inversion
INVERT_COLORS = True 

# ==================== DETECTOR INITIALIZATION ====================


def _build_aruco_dict_map():
    """Collect available OpenCV ArUco dictionary constants by name."""
    mapping = {}
    for name in dir(cv2.aruco):
        if not name.startswith("DICT_"):
            continue
        value = getattr(cv2.aruco, name)
        if isinstance(value, int):
            mapping[name] = value
    return dict(sorted(mapping.items(), key=lambda item: item[0]))


ARUCO_DICTIONARIES = _build_aruco_dict_map()


def get_aruco_dictionary_names():
    """Return all available ArUco dictionary names for UI dropdowns."""
    return list(ARUCO_DICTIONARIES.keys())


def get_aruco_dict_value(dict_name):
    """Resolve dictionary name (for example, DICT_4X4_50) to OpenCV value."""
    return ARUCO_DICTIONARIES.get(dict_name)


def get_aruco_dict_name(dict_value):
    """Resolve OpenCV dictionary value back to a dictionary name."""
    for name, value in ARUCO_DICTIONARIES.items():
        if value == dict_value:
            return name
    return None


def get_default_runtime_settings():
    """Return default runtime settings derived from config constants."""
    dict_name = get_aruco_dict_name(DICT_TYPE) or "DICT_4X4_50"
    return {
        "dict_type": dict_name,
        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,
        "square_length": SQUARE_LENGTH,
        "marker_length": MARKER_LENGTH,
        "invert_colors": INVERT_COLORS,
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
    }


def validate_runtime_settings(settings):
    """
    Validate and normalize runtime settings.

    Returns:
        tuple: (is_valid, message, normalized_settings)
    """
    defaults = get_default_runtime_settings()
    if not isinstance(settings, dict):
        return False, "Settings data must be a JSON object.", defaults

    normalized = dict(defaults)
    normalized.update(settings)

    dict_name = str(normalized.get("dict_type", defaults["dict_type"]))
    if dict_name not in ARUCO_DICTIONARIES:
        return False, f"Unknown dictionary: {dict_name}", defaults

    try:
        normalized["squares_x"] = int(normalized["squares_x"])
        normalized["squares_y"] = int(normalized["squares_y"])
        normalized["square_length"] = float(normalized["square_length"])
        normalized["marker_length"] = float(normalized["marker_length"])
        normalized["frame_width"] = int(normalized["frame_width"])
        normalized["frame_height"] = int(normalized["frame_height"])
        normalized["invert_colors"] = bool(normalized["invert_colors"])
    except (TypeError, ValueError, KeyError):
        return False, "One or more settings values are invalid.", defaults

    if normalized["squares_x"] < 2 or normalized["squares_y"] < 2:
        return False, "Board squares must be at least 2 in each direction.", defaults

    if normalized["square_length"] <= 0 or normalized["marker_length"] <= 0:
        return False, "Square and marker lengths must be positive.", defaults

    if normalized["marker_length"] >= normalized["square_length"]:
        return False, "Marker length must be smaller than square length.", defaults

    if normalized["frame_width"] <= 0 or normalized["frame_height"] <= 0:
        return False, "Frame width and height must be positive.", defaults

    normalized["dict_type"] = dict_name
    return True, "OK", normalized


def get_resolution_adaptive_rms_threshold(frame_width, frame_height):
    """
    Calculate resolution-adaptive RMS threshold for calibration.
    
    RMS is measured in pixels. Higher resolution cameras report proportionally larger
    pixel errors for the same physical detection accuracy. This function scales the
    RMS threshold based on resolution relative to a reference (1280×960).
    
    Args:
        frame_width (int): Camera frame width in pixels
        frame_height (int): Camera frame height in pixels
    
    Returns:
        float: Adaptive RMS threshold (pixels)
    
    Examples:
        1280×960 (reference):  ~1.5 pixels
        1920×1440:             ~2.0 pixels
        3840×2160 (4K):        ~2.8 pixels
    """
    # Reference resolution (baseline from default config)
    REF_WIDTH = 1280
    REF_HEIGHT = 960
    BASE_THRESHOLD = 1.5
    
    # Calculate image diagonals to handle arbitrary aspect ratios
    ref_diagonal = (REF_WIDTH**2 + REF_HEIGHT**2) ** 0.5
    current_diagonal = (frame_width**2 + frame_height**2) ** 0.5
    
    # Scale threshold proportionally
    resolution_factor = current_diagonal / ref_diagonal
    adaptive_threshold = BASE_THRESHOLD * resolution_factor
    
    return adaptive_threshold


def create_aruco_board(dict_name, squares_x, squares_y, square_length, marker_length):
    """Create and return a ChArUco board using runtime parameters."""
    dict_value = get_aruco_dict_value(dict_name)
    if dict_value is None:
        raise RuntimeError(f"Unknown ArUco dictionary: {dict_name}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_value)
    board = cv2.aruco.CharucoBoard(
        (int(squares_x), int(squares_y)),
        float(square_length),
        float(marker_length),
        aruco_dict,
    )
    return board


def create_detectors(dict_name, squares_x, squares_y, square_length, marker_length):
    """
    Initialize and return ArUco and ChArUco detectors using runtime parameters.

    Returns:
        tuple: (charuco_detector, aruco_detector, board, aruco_dict)
    """
    try:
        dict_value = get_aruco_dict_value(dict_name)
        if dict_value is None:
            raise RuntimeError(f"Unknown ArUco dictionary: {dict_name}")

        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_value)
        board = cv2.aruco.CharucoBoard(
            (int(squares_x), int(squares_y)),
            float(square_length),
            float(marker_length),
            aruco_dict,
        )

        params = cv2.aruco.DetectorParameters()
        charuco_params = cv2.aruco.CharucoParameters()

        charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, params)
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        return charuco_detector, aruco_detector, board, aruco_dict

    except Exception as e:
        raise RuntimeError(f"Failed to initialize detectors: {str(e)}")

def get_aruco_board():
    """
    Create and return the ChArUco board object.
    
    Returns:
        cv2.aruco.CharucoBoard: Configured ChArUco board
    """
    defaults = get_default_runtime_settings()
    return create_aruco_board(
        defaults["dict_type"],
        defaults["squares_x"],
        defaults["squares_y"],
        defaults["square_length"],
        defaults["marker_length"],
    )


def get_detectors():
    """
    Initialize and return ArUco and ChArUco detectors.
    
    Returns:
        tuple: (charuco_detector, aruco_detector, board, aruco_dict)
    
    Raises:
        RuntimeError: If detector initialization fails
    """
    defaults = get_default_runtime_settings()
    return create_detectors(
        defaults["dict_type"],
        defaults["squares_x"],
        defaults["squares_y"],
        defaults["square_length"],
        defaults["marker_length"],
    )
