import cv2
import numpy as np

def scale_img(img):
    if img is None:
        print("Error: Could not load image.")
    else:
        if len(img.shape) == 3 and img.shape[2] == 3:
            b, g, r = cv2.split(img)
        else:
            b = g = r = img

        b_n = cv2.normalize(b.astype(np.float32), None, 0, 65535, cv2.NORM_MINMAX)
        g_n = cv2.normalize(g.astype(np.float32), None, 0, 65535, cv2.NORM_MINMAX)
        r_n = cv2.normalize(r.astype(np.float32), None, 1, 65535, cv2.NORM_MINMAX)

        gray_16bit = np.clip((b_n + g_n + r_n) / 3.0, 0, 65535).astype(np.uint16)

        return gray_16bit

def undistort_image(img, k1, k2, k3, p1, p2, hfov_deg=92):
    orig_h, orig_w = img.shape[:2]
    new_w, new_h = 924, 924

    padded_img = np.zeros(
        (new_h, new_w) + (() if len(img.shape) == 2 else (img.shape[2],)),
        dtype=img.dtype,
    )
    x_offset = (new_w - orig_w) // 2
    y_offset = (new_h - orig_h) // 2
    padded_img[y_offset : y_offset + orig_h, x_offset : x_offset + orig_w] = img

    # Focal length must be based on original width (orig_w)
    hfov_rad = np.radians(hfov_deg)
    f = (orig_w / 2.0) / np.tan(hfov_rad / 2.0)

    # Principal point shifted to the center of the 924x924 canvas
    camera_matrix = np.array(
        [[f, 0, new_w / 2.0], [0, f, new_h / 2.0], [0, 0, 1.0]], dtype=np.float32
    )

    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    undistorted = cv2.undistort(padded_img, camera_matrix, dist_coeffs)
    return undistorted

def rotate_img(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, float(angle) - 90, 1.0)
    new_w, new_h = 924, 924
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(frame, M, (new_w, new_h))
    return rotated

def trapezium_scale(image, pitch, roll, hfov_deg=92):
    pitch_deg = float(pitch)
    roll_deg = float(roll)

    if pitch_deg > 180:
        pitch_deg -= 360
    if roll_deg > 180:
        roll_deg -= 360

    h, w = image.shape[:2]

    hfov_rad = np.radians(hfov_deg)
    f = (w / 2.0) / np.tan(hfov_rad / 2.0)

    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float32)

    pitch_rad = np.radians(-pitch_deg)
    roll_rad = np.radians(-roll_deg)

    Ry = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
    ], dtype=np.float32)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)],
    ], dtype=np.float32)

    R_undo = Ry @ Rx
    H = K @ R_undo @ np.linalg.inv(K)

    new_w, new_h = 924, 924

    translation = np.array([
        [1, 0, (new_w - w) / 2.0],
        [0, 1, (new_h - h) / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    H_centered = translation @ H

    corrected_image = cv2.warpPerspective(
        image,
        H_centered,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return corrected_image

def preprocess(img, k1, k2, k3, p1, p2, yaw, pitch, roll, alt):
    scaled = scale_img(img)
    undistorted = undistort_image(scaled, k1, k2, k3, p1, p2)
    rotated = rotate_img(undistorted, yaw)
    normalised = trapezium_scale(rotated, pitch, roll)
    return normalised