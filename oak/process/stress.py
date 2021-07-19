import pandas as pd
from oak.process.process_base import ProcessBase


class Stress(ProcessBase):
    name: str = "Stress"

    def update(self, stress_score: int):
        """Updates the stress analysis with new information.

        Parameters
        ----------
        stress_score : int
            Stress coeficent, can be 0 or 1.
        """

        # Add new score
        self.ser_score = self.ser_score.append(
            pd.Series([stress_score], index=[self.total_elements]),
        )

        self.total_elements += 1

    def restart_series(self):
        """Clean up the series and restart them with fewer values"""
        self.ser_score = self.ser_score.iloc[-int(len(self.ser_score) / 4) :]
