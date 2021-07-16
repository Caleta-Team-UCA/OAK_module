import numpy as np
import pandas as pd

from typing import Iterable, List
from oak.process.process_base import ProcessBase


class BreathConfig:
    # Width and height
    height = 480
    width = 640

    # We need to be accurate, so we use a very small ROI
    topLeft = {"x": 0.4, "y": 0.4}
    bottomRight = {"x": 0.42, "y": 0.42}

    # Size of the ROI
    width_roi = 20

    # Position dx and dy of the ROI
    dy = 0.9
    dx = 1.5

    xmin, ymin, xmax, ymax = 0, 0, 0, 0

    def get_roi_points(self):
        return self.topLeft, self.bottomRight

    def get_roi_bbox(self):
        return self.xmin, self.ymin, self.xmax, self.ymax


class Breath(ProcessBase):
    name: str = "Breath"

    def __init__(self, breath_config: BreathConfig):
        self.breath_config = breath_config

    def get_breath_config(self) -> BreathConfig:
        return self.breath_config

    def set_breath_config(self, breath_config: BreathConfig):
        self.breath_config = breath_config

    def restart_series(self):
        self.ser_score = self.ser_score.iloc[-int(len(self.ser_score) / 4) :]

    def _get_roi_coordinates(self, face_detections: Iterable[int]):
        # ROI coordinates
        # width_roi is the size of the ROI, user-defined
        # dx and dy can be change interactively using the WASD keys
        self.breath_config.xmin = int(
            face_detections[0]
            + self.breath_config.dx * (face_detections[2] - face_detections[0]) / 2
        ) - int(self.breath_config.width_roi / 2)
        self.breath_config.ymin = int(
            face_detections[3]
            + self.breath_config.dy * (face_detections[3] - face_detections[1])
        )
        self.breath_config.xmax = self.breath_config.xmin + self.breath_config.width_roi
        self.breath_config.ymax = self.breath_config.ymin + self.breath_config.width_roi

        # Section needed to catch some errors when the ROI is outside the frame. In such situations the ROI is kept inside
        if self.breath_config.xmin > 1:
            self.breath_config.topLeft["x"] = (
                self.breath_config.xmin / self.breath_config.width
            )
        else:
            self.breath_config.topLeft["x"] = 1 / self.breath_config.width
            self.breath_config.bottomRight["x"] = (
                self.breath_config.topLeft["x"] + self.breath_config.width_roi
            )

        if self.breath_config.ymin > 1:
            self.breath_config.topLeft["y"] = (
                self.breath_config.ymin / self.breath_config.height
            )
        else:
            self.breath_config.topLeft["y"] = 1 / self.breath_config.height
            self.breath_config.bottomRight["y"] = (
                self.breath_config.topLeft["y"] + self.breath_config.width_roi
            )

        if self.breath_config.xmax < self.breath_config.width:
            self.breath_config.bottomRight["x"] = (
                self.breath_config.xmax / self.breath_config.width
            )
        else:
            self.breath_config.bottomRight["x"] = (
                self.breath_config.width / self.breath_config.width
            )
            self.breath_config.topLeft["x"] = (
                self.breath_config.bottomRight["x"] - self.breath_config.width_roi
            )

        if self.breath_config.ymax < self.breath_config.height:
            self.breath_config.bottomRight["y"] = (
                self.breath_config.ymax / self.breath_config.height
            )
        else:
            self.breath_config.bottomRight["y"] = (
                self.breath_config.height / self.breath_config.height
            )
            self.breath_config.topLeft["y"] = (
                self.breath_config.bottomRight["y"] - self.breath_config.width_roi
            )

    def _get_depth_roi(self, calculator_results: List[int]) -> int:
        # Measure depth from stereo-matching between left-right cameras and adds the value to the variable z
        return (
            int(calculator_results[len(calculator_results) - 1].spatialCoordinates.z)
            / 10
        )

    def update(
        self,
        face_detections: Iterable[int],
        depth_frame: np.array,
        calculator_results: List[int],
    ):
        if face_detections is not None:
            self._get_roi_coordinates(face_detections)

        if face_detections is not None and calculator_results is not None:
            distance = self._get_depth_roi(calculator_results)
            print("Distance: " + str(distance))

            self.ser_score = self.ser_score.append(
                pd.Series([distance], index=[self.total_elements])
            )

            self.total_elements += 1
