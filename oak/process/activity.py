from typing import Iterable

import numpy as np
import pandas as pd
from oak.process.process_base import ProcessBase


class Activity(ProcessBase):
    status: list = [0, 0, 0, 0]  # arm left, right, leg left, right
    name: str = "Activity"

    def __init__(
        self,
    ):
        """Performs the pose analysis. Detects the move of each limb

        Parameters
        ----------
        size : int, optional
            Length of the stored series, by default 1000
        frequency : int, optional
            Rate at which plots are updated, in frames, by default 24
        """
        # Initialize series of box dimensions
        self.ser_right = pd.Series(name="Right")
        self.ser_left = pd.Series(name="Left")
        self.ser_up = pd.Series(name="Up")
        self.ser_down = pd.Series(name="Down")

    def restart_series(self):
        """Clean up the series and restart them with fewer values"""
        size = -int(len(self.ser_right) / 4)
        # Initialize series of box dimensions
        self.ser_right = self.ser_right.iloc[size:]
        self.ser_left = self.ser_left.iloc[size:]
        self.ser_up = self.ser_up.iloc[size:]
        self.ser_down = self.ser_down.iloc[size:]
        # Initialize pd.series of score
        self.ser_score = self.ser_score.iloc[size:]

    def _update_size_series(
        self,
        body_detections: Iterable[int],
        face_detections: Iterable[int],
    ):
        """Updates the series with the last detections

        Parameters
        ----------
        body_detections : list
            List of body detections [xmin, ymin, xmax, ymax]
        face_detections : list
            List of face detections [xmin, ymin, xmax, ymax]
        """
        # Check if there are any body_detections
        try:
            # Compute the face size
            size = face_detections[3] - face_detections[1]
            right = (body_detections[2] - face_detections[2]) / size
            left = (face_detections[0] - body_detections[0]) / size
            up = (face_detections[1] - body_detections[1]) / size
            down = (body_detections[3] - face_detections[3]) / size
        except (IndexError, TypeError) as er:
            # If not, the size is the last value
            right = self.ser_right.iloc[-1]
            left = self.ser_left.iloc[-1]
            up = self.ser_up.iloc[-1]
            down = self.ser_down.iloc[-1]
        # Update the series of box sizes
        self.ser_right = self.ser_right.append(
            pd.Series([right], index=[self.total_elements])
        )
        self.ser_left = self.ser_left.append(
            pd.Series([left], index=[self.total_elements])
        )
        self.ser_up = self.ser_up.append(pd.Series([up], index=[self.total_elements]))
        self.ser_down = self.ser_down.append(
            pd.Series([down], index=[self.total_elements])
        )

    @property
    def arm_right(self):
        ser = np.abs(self.ser_right)
        ser[ser > 1] = 1
        return ser

    @property
    def arm_left(self):
        ser = np.abs(self.ser_left)
        ser[ser > 1] = 1
        return ser

    @property
    def legs(self):
        ser = self.ser_down - 2.5
        ser[ser < 0] = 0
        ser[ser > 1] = 1
        return ser

    @property
    def dict_series(self) -> dict:
        return {
            "Right arm": self.arm_right,
            "Left arm": self.arm_left,
            "Legs": self.legs,
        }

    @property
    def dict_scores(self) -> dict:
        return {
            "left": self.arm_left.mean(),
            "right": self.arm_right.mean(),
            "down": self.legs.mean(),
        }

    def _update_score_series(self):
        """Updates score status"""
        # Store the previous status
        status_before = self.status.copy()
        # Update the status
        self.status[0] = 0 if self.arm_left.iloc[-1] < 0.6 else 1
        self.status[1] = 0 if self.arm_right.iloc[-1] < 0.6 else 1
        self.status[2] = 0 if self.legs.iloc[-1] < 3.1 else 1
        self.status[3] = 0 if self.legs.iloc[-1] < 3.1 else 1
        # Compute the mobility score as the number of status that have changed
        mob_score = np.mean(np.abs(np.array(self.status) - np.array(status_before)))
        self.ser_score = self.ser_score.append(
            pd.Series([mob_score], index=[self.total_elements])
        )

        self.total_elements += 1

    def update(
        self,
        body_detections: Iterable[int],
        face_detections: Iterable[int],
    ):
        """Updates the analysis with new information

        Parameters
        ----------
        body_detections : list
            List of body detections [xmin, ymin, xmax, ymax]
        face_detections : list
            List of face detections [xmin, ymin, xmax, ymax]
        """
        # Update the series
        self._update_size_series(body_detections, face_detections)
        # Update score
        self._update_score_series()
