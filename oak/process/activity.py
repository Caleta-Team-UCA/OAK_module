import json
from typing import Iterable

import numpy as np
from oak.utils.series import PlotSeries, Series


class Activity:
    status: list = [0, 0, 0, 0]  # arm left, right, leg left, right
    _timer: int = 0

    def __init__(
        self,
        size: int = 240,
        frequency: int = 12,
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
        self.ser_right = Series(size=size, frequency=frequency, label="Right")
        self.ser_left = Series(size=size, frequency=frequency, label="Left")
        self.ser_up = Series(size=size, frequency=frequency, label="Up")
        self.ser_down = Series(size=size, frequency=frequency, label="Down")
        # Initialize series of score
        self.ser_score = Series(size=size, frequency=frequency, label="Mob. score")

        # Store update frequency, initialize timer
        self.frequency = frequency
        # Plot series
        self.plot_series = PlotSeries(
            [self.ser_score],
            xlim=(0, size),
            ylim=(0, 1),
        )

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
            right = self.ser_right.ser[-1]
            left = self.ser_left.ser[-1]
            up = self.ser_up.ser[-1]
            down = self.ser_down.ser[-1]
        # Update the series of box sizes
        self.ser_right.append(right)
        self.ser_left.append(left)
        self.ser_up.append(up)
        self.ser_down.append(down)

    def _update_score_series(self):
        """Updates score status"""
        # Store the previous status
        status_before = self.status.copy()
        # Get the last values of each series
        right = self.ser_right[-1]
        left = self.ser_left[-1]
        down = self.ser_down[-1]
        if down < 3.1:
            self.status[2] = 0
            self.status[3] = 0
        else:
            self.status[2] = 1
            self.status[3] = 1
        if left < 0.6:
            self.status[0] = 0
        else:
            self.status[0] = 1
        if right < 0.6:
            self.status[1] = 0
        else:
            self.status[1] = 1
        # Compute the mobility score as the number of status that have changed
        mob_score = np.mean(np.abs(np.array(self.status) - np.array(status_before)))
        self.ser_score.append(mob_score)

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
        # Plot the evolution of box size
        self._timer += 1
        if self._timer >= self.frequency:
            self.plot_series.update(method="max")
            self._timer = 0
