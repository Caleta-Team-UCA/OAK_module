import depthai as dai
import numpy as np
import pandas as pd

from typing import Iterable, List
from oak.process.process_base import ProcessBase


class BreathConfig():
    # Width and height
    height = 480
    width = 640

    # We need to be accurate, so we use a very small ROI
    topLeft = dai.Point2f(0.4, 0.4)
    bottomRight = dai.Point2f(0.42, 0.42)

    # Size of the ROI
    width_roi = 20

    # Position dx and dy of the ROI
    dy = 0.9
    dx = 1.5

    x1, x2, y1, y2 = 0, 0, 0, 0


class Breath(ProcessBase):
    name: str = "Breath"

    def __init__(self, breath_config: BreathConfig):
        self.breath_config = breath_config
        self.ser_distance = pd.Series(name="Chest distance")

    def get_breath_config(self) -> BreathConfig:
        return self.breath_config

    def set_breath_config(self, breath_config: BreathConfig):
        self.breath_config = breath_config

    def restart_series(self):
        self.ser_distance = self.ser_distance.iloc[-int(len(self.ser_distance) / 4) :]

    def _get_depth_located_roi(
        self,
        face_detections: Iterable[int],
        depth_frame: np.array,
        calculator_results: List[int]
    ):
        for depthData in calculator_results:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depth_frame.shape[1], height=depth_frame.shape[0])

            # ROI coordinates
            # width_roi is the size of the ROI, user-defined
            # dx and dy can be change interactively using the WASD keys
            xmin = int(self.breath_config.x1 + self.breath_config.dx * (self.breath_config.x2 - self.breath_config.x1) / 2) - int(self.breath_config.width_roi / 2)
            ymin = int(self.breath_config.y2 + self.breath_config.dy * (self.breath_config.y2 - self.breath_config.y1))
            xmax = xmin + self.breath_config.width_roi
            ymax = ymin + self.breath_config.width_roi

            # Section needed to catch some errors when the ROI is outside the frame. In such situations the ROI is kept inside
            if xmin > 1:
                self.breath_config.topLeft.x = xmin / self.breath_config.width
            else:
                self.breath_config.topLeft.x = 1 / self.breath_config.width
                self.breath_config.bottomRight.x = self.breath_config.topLeft.x + self.breath_config.width_roi

            if ymin > 1:
                self.breath_config.topLeft.y = ymin / self.breath_config.height
            else:
                self.breath_config.topLeft.y = 1 / self.breath_config.height
                self.breath_config.bottomRight.y = self.breath_config.topLeft.y + self.breath_config.width_roi

            if xmax < width:
                self.breath_config.bottomRight.x = xmax / self.breath_config.width
            else:
                self.breath_config.bottomRight.x = self.breath_config.width / self.breath_config.width
                self.breath_config.topLeft.x = self.breath_config.bottomRight.x - self.breath_config.width_roi

            if ymax < self.breath_config.height:
                self.breath_config.bottomRight.y = ymax / self.breath_config.height
            else:
                self.breath_config.bottomRight.y = self.breath_config.height / self.breath_config.height
                self.breath_config.topLeft.y = self.breath_config.bottomRight.y - self.breath_config.width_roi

    def _get_respiration_depth(
        self,
        face_detections: Iterable[int]
    ) -> int:
        # Denormalization of the bounding box coordinates
        self.breath_config.x1 = int(face_detections[0] * self.breath_config.width)
        self.breath_config.x2 = int(face_detections[2] * self.breath_config.width)
        self.breath_config.y1 = int(face_detections[1] * self.breath_config.height)
        self.breath_config.y2 = int(face_detections[3] * self.breath_config.height)

        # Measure depth from stereo-matching between left-right cameras and adds the value to the variable z
        return int(depthData.spatialCoordinates.z)/10

    def update(
        self,
        face_detections: Iterable[int],
        depth_frame: np.array,
        calculator_results: List[int]
    ):
        self._get_depth_located_roi(face_detections, depth_frame, calculator_results)
        distance = self._get_respiration_depth(face_detections)

        self.ser_distance = self.ser_distance.append(
            pd.Series([distance], index=[self.total_elements])
        )

        self.total_elements += 1
