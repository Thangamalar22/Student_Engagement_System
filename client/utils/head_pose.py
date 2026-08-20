# client/utils/head_pose.py
# -------------------------------------------------------
# Head pose estimation using OpenCV's solvePnP.
#
# We give solvePnP:
#   - A generic 3D face model (6 points in mm)
#   - The corresponding 2D landmark positions in the current frame
#   - The camera intrinsic matrix (approximated from frame size)
#
# It returns a rotation vector; we convert to Euler angles
# (pitch, yaw, roll) in degrees.
#
# Convention used:
#   pitch: +down / -up
#   yaw  : +right / -left  (from student's perspective)
#   roll : +tilt right / -tilt left
# -------------------------------------------------------

import cv2
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config import HEAD_POSE_3D_MODEL, POSE_LANDMARK_IDS


# Cache the 3D model as a numpy array once
MODEL_POINTS_3D = np.array(HEAD_POSE_3D_MODEL, dtype=np.float64)


def _build_camera_matrix(img_w: int, img_h: int) -> np.ndarray:
    """
    Approximate camera intrinsic matrix using frame size.
    We don't have real calibration data per webcam, so we assume:
      - focal length ≈ image width (typical for webcams)
      - principal point at image center
    This is a well-known approximation; error is small for gaze/pose.
    """
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    return np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1        ]
    ], dtype=np.float64)


def _rotation_vector_to_euler(rvec: np.ndarray) -> tuple:
    """
    Convert OpenCV rotation vector -> rotation matrix -> Euler angles (deg).
    """
    rmat, _ = cv2.Rodrigues(rvec)

    # Decompose rotation matrix into Euler angles
    # Using ZYX (yaw, pitch, roll) convention
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw   = np.arctan2(-rmat[2, 0], sy)
        roll  = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw   = np.arctan2(-rmat[2, 0], sy)
        roll  = 0.0

    return (np.degrees(pitch), np.degrees(yaw), np.degrees(roll))


def estimate_head_pose(landmarks, img_w: int, img_h: int) -> dict:
    """
    Run solvePnP on the 6 designated face landmarks.

    Returns
    -------
    dict with:
        pitch, yaw, roll : floats in degrees (or None if solve failed)
        success          : bool
        image_points     : the 2D points used (for optional drawing)
        rvec, tvec       : raw solvePnP output (for drawing 3D axes)
    """
    # Collect 2D image points corresponding to the 3D model
    image_points_2d = []
    for idx in POSE_LANDMARK_IDS:
        lm = landmarks.landmark[idx]
        image_points_2d.append([lm.x * img_w, lm.y * img_h])
    image_points_2d = np.array(image_points_2d, dtype=np.float64)

    camera_matrix = _build_camera_matrix(img_w, img_h)
    dist_coeffs   = np.zeros((4, 1))  # assume no lens distortion

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return {"success": False, "pitch": None, "yaw": None, "roll": None,
                "image_points": image_points_2d, "rvec": None, "tvec": None}

    pitch, yaw, roll = _rotation_vector_to_euler(rvec)

    return {
        "success": True,
        "pitch": pitch, "yaw": yaw, "roll": roll,
        "image_points": image_points_2d,
        "rvec": rvec, "tvec": tvec,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs
    }


def draw_pose_axes(frame: np.ndarray, pose_result: dict,
                    axis_length: float = 60.0) -> None:
    """
    Draw a 3D coordinate axis on the nose to visualize head pose.
    Red = X (right), Green = Y (down), Blue = Z (out of face).
    """
    if not pose_result["success"]:
        return

    axis_3d = np.float32([
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ]).reshape(-1, 3)

    projected, _ = cv2.projectPoints(
        axis_3d, pose_result["rvec"], pose_result["tvec"],
        pose_result["camera_matrix"], pose_result["dist_coeffs"]
    )

    nose_2d = tuple(pose_result["image_points"][0].astype(int))
    x_axis = tuple(projected[0].ravel().astype(int))
    y_axis = tuple(projected[1].ravel().astype(int))
    z_axis = tuple(projected[2].ravel().astype(int))

    cv2.line(frame, nose_2d, x_axis, (0, 0, 255), 2)   # X: red
    cv2.line(frame, nose_2d, y_axis, (0, 255, 0), 2)   # Y: green
    cv2.line(frame, nose_2d, z_axis, (255, 0, 0), 2)   # Z: blue