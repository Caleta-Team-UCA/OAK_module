import numpy as np
import pandas as pd

from typing import Iterable, List
from oak.process.process_base import ProcessBase


class Breath(ProcessBase):
    name: str = "Breath"

    # We need to be accurate, so we use a very small ROI
    topLeft: dict[str, float] = {"x": 0.4, "y": 0.4}
    bottomRight: dict[str, float] = {"x": 0.42, "y": 0.42}

    # Size of the ROI
    width_roi: int = 0.05

    # Position dx and dy of the ROI
    dy: float = 0.4
    dx: float = 1.5

    xmin: int = 0
    ymin: int = 0
    xmax: int = 0
    ymax: int = 0

    def restart_series(self):
        self.ser_score = self.ser_score.iloc[-int(len(self.ser_score) / 4) :]

    def _get_roi_coordinates(self, face_detections: Iterable[int]):
        x1, y1, x2, y2 = face_detections
        # ROI coordinates
        # width_roi is the size of the ROI, user-defined
        # dx and dy can be change interactively using the WASD keys
        xmin = x1 + self.dx * (x2 - x1) / 2 - self.width_roi / 2
        ymin = y2 + self.dy * (y2 - y1)
        xmax = xmin + self.width_roi
        ymax = ymin + self.width_roi

        print(face_detections, (xmin, ymin, xmax, ymax))

        # Section needed to catch some errors when the ROI is outside the frame. In such situations the ROI is kept inside
        if xmin > 0:
            self.topLeft["x"] = xmin
        else:
            self.topLeft["x"] = 0
            self.bottomRight["x"] = self.topLeft["x"] + self.width_roi

        if ymin > 0:
            self.topLeft["y"] = ymin
        else:
            self.topLeft["y"] = 0
            self.bottomRight["y"] = self.topLeft["y"] + self.width_roi

        if xmax < 1:
            self.bottomRight["x"] = xmax
        else:
            self.bottomRight["x"] = 1
            self.topLeft["x"] = self.bottomRight["x"] - self.width_roi

        if ymax < 1:
            self.bottomRight["y"] = ymax
        else:
            self.bottomRight["y"] = 1
            self.topLeft["y"] = self.bottomRight["y"] - self.width_roi

    def _get_depth_roi(self, calculator_results: List[int]) -> int:
        # Measure depth from stereo-matching between left-right cameras and adds the value to the variable z
        return (
            int(calculator_results[len(calculator_results) - 1].spatialCoordinates.z)
            / 10
        )

    @property
    def get_roi_corners(self) -> tuple[float]:
        return (
            self.topLeft["x"],
            self.topLeft["y"],
            self.bottomRight["x"],
            self.bottomRight["y"],
        )

    def update(
        self,
        face_detections: Iterable[int],
        calculator_results: List[int],
    ):

        if face_detections is not None:
            self._get_roi_coordinates(face_detections)

        if face_detections is not None and calculator_results is not None:
            distance = self._get_depth_roi(calculator_results)

            self.ser_score = self.ser_score.append(
                pd.Series([distance], index=[self.total_elements])
            )

            self.total_elements += 1
