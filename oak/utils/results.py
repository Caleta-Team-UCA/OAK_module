from typing import List

import matplotlib.pyplot as plt

COLORS = ["salmon", "cornflowerblue", "forestgreen"]


class PlotSeries:
    def __init__(self, list_ser: List):
        """Plots a list of Series on a figure, that can be updated on real time

        Parameters
        ----------
        list_ser : list
            List of neocam.utils.series Series
        xlim : tuple, optional
            Limits of X axis, by default None
        ylim : tuple, optional
            Limits of Y axis, by default None
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

            ax.plot(
                ser.score.index.to_numpy(),
                ser.get_moving_average(),
                self.ser_plot_data[ser.name]["color"],
                label=ser.name,
            )

            ax.grid()
            ax.set_xlabel("Time (frames)")
            ax.set_ylabel(ser.name)

        plt.tight_layout()
        plt.show(block=False)

    def update(self, method: str = None):
        """Updates the figure"""
        for name, ser_data in self.ser_plot_data.items():
            ser = ser_data["object"]
            moving_mean = ser.score.rolling(window=10).mean().to_numpy()
            ser_data["ax"].plot(
                ser.score.index.to_numpy(),
                moving_mean,
                ser_data["color"],
                label=name,
            )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Closes the figure"""
        plt.close(self.fig)
