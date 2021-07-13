from oak.utils.series import Series


class Stress:
    def __init__(
        self,
        size: int = 240,
        frequency: int = 12,
    ):

        self.ser_score = Series(size=size, frequency=frequency, label="Stress score")

    def update(self, stress_score: int):
        """Updates the stress analysis with new information.

        Parameters
        ----------
        stress_score : int
            Stress coeficent, can be 0 or 1.
        """

        # Add new score
        self.ser_score.append(stress_score)
