from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def undistort(filename, images_dir, metadata_csv=None):
    """Undistort one drone image and return it as a NumPy array (same size and dtype as the file)."""
    images_dir = Path(images_dir)
    metadata_csv = metadata_csv or images_dir / "metadata.csv"

    meta = pd.read_csv(metadata_csv).set_index("filename")
    r = meta.loc[filename]

    img = cv2.imread(str(images_dir / filename), cv2.IMREAD_UNCHANGED)  # keeps 16-bit
    if img is None:
        raise FileNotFoundError(images_dir / filename)
    h, w = img.shape[:2]

    K = np.array([[r.cam_fx, 0, r.cam_cx],
                  [0, r.cam_fy, r.cam_cy],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array([r.cam_k1, r.cam_k2, r.cam_p1, r.cam_p2, r.cam_k3], dtype=np.float64)

    # Zoom in just enough to remove black edges, using the same zoom in x and y (no stretching).
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
    s = max(newK[0, 0] / K[0, 0], newK[1, 1] / K[1, 1])
    newK[0, 0], newK[1, 1] = K[0, 0] * s, K[1, 1] * s

    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_32FC1)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
