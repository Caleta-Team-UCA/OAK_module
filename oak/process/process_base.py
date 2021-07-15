from abc import abstractmethod
import pandas as pd


class ProcessBase:
    total_elements: int = 0
    ser_score: pd.Series = pd.Series(name="Score")

    @abstractmethod
    def restart_series(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def get_moving_average(self):
        return (
            self.ser_score.rolling(window=int(len(self.ser_score) / 6))
            .mean()
            .to_numpy()
        )

    @property
    def score(self):
        return self.ser_score
