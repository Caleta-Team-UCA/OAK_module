from typing import List

import matplotlib.pyplot as plt

COLORS = ["salmon", "cornflowerblue", "forestgreen"]


class PlotSeries:
    def __init__(self, list_ser: List):
        """Plots studies made on a figure, that can be updated on real time

        Parameters
        ----------
        list_ser : list
            List of ProcessBase objects
        """
        self.fig = plt.figure()

        self.ser_plot_data = {}
        for i, ser in enumerate(list_ser):
            c = COLORS[i]
            ax = plt.subplot(int(f"{len(list_ser)}1{i + 1}"))
            self.fig.canvas.draw()

            self.ser_plot_data[ser.name] = {}
            self.ser_plot_data[ser.name]["object"] = ser
            self.ser_plot_data[ser.name]["color"] = c
            self.ser_plot_data[ser.name]["ax"] = ax

            self._plot_ser(ser, ax)

            ax.grid()
            ax.set_xlabel("Time (frames)")
            ax.set_ylabel(ser.name)

        plt.tight_layout()
        plt.show(block=False)

    def _plot_ser(self, ser, ax):
        moving_mean = ser.get_moving_average()
        if type(moving_mean) == dict:
            for i, (name, ser_sub) in enumerate(moving_mean.items()):
                ax.plot(
                    ser.score.index.to_numpy(),
                    ser_sub,
                    COLORS[i],
                    label=name,
                )
            ax.legend(moving_mean.keys())
        else:
            ax.plot(
                ser.score.index.to_numpy(),
                moving_mean,
                self.ser_plot_data[ser.name]["color"],
                label=ser.name,
            )

    def update(self, method: str = None):
        """Updates the figure"""
        for name, ser_data in self.ser_plot_data.items():
            self._plot_ser(ser_data["object"], ser_data["ax"])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Closes the figure"""
        plt.close(self.fig)
