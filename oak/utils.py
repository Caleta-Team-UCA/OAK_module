import cv2
from time import monotonic
import depthai as dai
import numpy as np


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


def process_frame(frame: np.array, width: int, height: int) -> dai.ImgFrame:
    # Generate ImgFrame to use as input of the Pipeline
    img = dai.ImgFrame()
    img.setData(to_planar(frame, (width, height)))
    img.setTimestamp(monotonic())
    img.setWidth(width)
    img.setHeight(height)

    return img


def frame_norm(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Normalizes the frame"""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)
