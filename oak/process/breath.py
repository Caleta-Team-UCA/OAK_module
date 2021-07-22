from typing import Iterable, List, Dict, Tuple

import pandas as pd
from oak.process.process_base import ProcessBase
from scipy.signal import find_peaks


class Breath(ProcessBase):
    name: str = "Breath"

    # Width and height
    height: int = 300
    width: int = 300

    # We need to be accurate, so we use a very small ROI
    topLeft: Dict[str, float] = {"x": 0.4, "y": 0.4}
    bottomRight: Dict[str, float] = {"x": 0.42, "y": 0.42}

    # Size of the ROI
    width_roi: int = 20

    # Position dx and dy of the ROI
    dy: float = 0.5
    dx: float = 1.3

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
        self.xmin = int(x1 + self.dx * (x2 - x1) / 2) - int(self.width_roi / 2)

        self.ymin = int(y2 + self.dy * (y2 - y1))
        self.xmax = self.xmin + self.width_roi
        self.ymax = self.ymin + self.width_roi

        # Section needed to catch some errors when the ROI is outside the frame. In such situations the ROI is kept inside
        if self.xmin > 1:
            self.topLeft["x"] = self.xmin / self.width
        else:
            self.topLeft["x"] = 1 / self.width
            self.bottomRight["x"] = self.topLeft["x"] + self.width_roi

        if self.ymin > 1:
            self.topLeft["y"] = self.ymin / self.height
        else:
            self.topLeft["y"] = 1 / self.height
            self.bottomRight["y"] = self.topLeft["y"] + self.width_roi

        if self.xmax < self.width:
            self.bottomRight["x"] = self.xmax / self.width
        else:
            self.bottomRight["x"] = self.width / self.width
            self.topLeft["x"] = self.bottomRight["x"] - self.width_roi

        if self.ymax < self.height:
            self.bottomRight["y"] = self.ymax / self.height
        else:
            self.bottomRight["y"] = self.height
            self.topLeft["y"] = self.bottomRight["y"] - self.width_roi

    def _get_depth_roi(self, calculator_results: List[int]) -> int:
        # Measure depth from stereo-matching between left-right cameras and adds the value to the variable z
        return (
            int(calculator_results[len(calculator_results) - 1].spatialCoordinates.z)
            / 10
        )

    @property
    def get_roi_corners(self) -> Tuple[float]:
        return (
            self.topLeft["x"],
            self.topLeft["y"],
            self.bottomRight["x"],
            self.bottomRight["y"],
        )

    def update(
        self,
        face_detections: Iterable[int],
        frame_shape: Tuple[int],
        calculator_results: List[int],
    ):
        self.width = frame_shape[1]
        self.height = frame_shape[0]

        if face_detections is not None:
            self._get_roi_coordinates(face_detections)

        if face_detections is not None and calculator_results is not None:
            distance = self._get_depth_roi(calculator_results)
            # print("Distance: " + str(distance))

            self.ser_score = self.ser_score.append(
                pd.Series([distance], index=[self.total_elements])
            )

            self.total_elements += 1

    def get_bpm(self, delay):
        peak_indices, _ = find_peaks(-1 * self.ser_score.to_numpy(), prominence=0.3)
        peak_count = len(peak_indices)
        return peak_count * 60 / delay
