from abc import abstractmethod

import numpy as np
import pandas as pd


class ProcessBase:
    total_elements: int = 0
    ser_score: pd.Series = pd.Series(name="Score")

    @abstractmethod
    def restart_series(self):
        """Clean up the series and restart them with fewer values"""
        pass

    @abstractmethod
    def update(self):
        """Updates the analysis with new information."""
        pass

    def get_moving_average(self) -> np.array:
        """Get moving average of the scores"""
        return (
            self.ser_score.rolling(window=int(len(self.ser_score) / 6))
            .mean()
            .to_numpy()
        )

    def get_dict_series(self) -> dict:
        """Returns a dictionary of series to plot"""
        return {"movavg": self.get_moving_average()}

    @property
    def score(self):
        """Get score series."""
        return self.ser_score
