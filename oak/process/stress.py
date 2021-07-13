from oak.utils.series import PlotSeries, Series

class Stress:
    _timer: int = 0

    def __init__(self,
        size: int = 240,
        frequency: int = 12,):
    
        self.ser_score = Series(size = size, frequency=frequency, label = "Stress score")

        # Store update frequency, initialize timer
        self.frequency = frequency
        # Plot series
        self.plot_series = PlotSeries(
            [self.ser_score],
            xlim=(0, size),
            ylim=(0, 1),
        )

    def update(self, stress_score: int):
        """Updates the stress analysis with new information.

        Parameters
        ----------
        stress_score : int
            Stress coeficent, can be 0 or 1.
        """

        # Add new score
        self.ser_score.append(stress_score)
        
        # Plot evolution
        self._timer += 1
        if self._timer >= self.frequency:
            self.plot_series.update(method="movavg")
            self._timer = 0