from typing import List, Iterable
import numpy as np
import cv2

import matplotlib.pyplot as plt
from oak.process.process_base import ProcessBase

COLORS = ["salmon", "cornflowerblue", "forestgreen"]


class PlotSeries:
    def __init__(self, list_proc: Iterable[ProcessBase]):
        """Plots studies made on a figure, that can be updated on real time

        Parameters
        ----------
        list_proc : Iterable[ProcessBase]
            List of ProcessBase objects
        """
        self.fig = plt.figure()

        self.proc_plot_data = {}
        for i, proc in enumerate(list_proc):
            ax = plt.subplot(int(f"{len(list_proc)}1{i + 1}"))
            self.fig.canvas.draw()

            self.proc_plot_data[proc.name] = {}
            self.proc_plot_data[proc.name]["object"] = proc
            # self.proc_plot_data[proc.name]["color"] = COLORS[i]
            self.proc_plot_data[proc.name]["ax"] = ax

            self._plot_process(proc, ax)

            ax.grid()
            ax.set_xlabel("Time (frames)")
            ax.set_ylabel(proc.name)

        plt.tight_layout()
        # plt.show(block=False)

    def _plot_process(self, proc: ProcessBase, ax):
        """Plots a process in given axis"""
        dict_plot = proc.dict_series
        for i, (name, ser_sub) in enumerate(dict_plot.items()):
            ax.plot(
                proc.score.index.to_numpy(),
                ser_sub,
                COLORS[i],
                label=name,
            )
        keys = dict_plot.keys()
        if len(keys) > 1:
            ax.legend(keys)

    def update(self, method: str = None):
        """Updates the figure"""
        for name, ser_data in self.proc_plot_data.items():
            self._plot_process(ser_data["object"], ser_data["ax"])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype="uint8")

        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def close(self):
        """Closes the figure"""
        plt.close(self.fig)
