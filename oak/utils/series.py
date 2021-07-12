import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


def moving_average(ser: np.array, size: int = 24) -> np.array:
    """Computes the moving average of a series, returns a series of same length as the
    original (first elements remain the same)

    Parameters
    ----------
    ser : numpy.array
    size : int

    Returns
    -------
    numpy.array
    """
    ret = np.convolve(ser, np.ones(size), "valid") / size
    return np.append(np.repeat(np.nan, size - 1), ret)


class Series:
    line: Line2D = None

    def __init__(self, size: int = 1000, frequency: int = 24, label: str = ""):
        """Series

        Parameters
        ----------
        size : int, optional
            Length of the series, by default 1000
        frequency : int, optional
            Size of the moving average, by default 24
        label : str, optional
            Name of the series, by default ""
        """
        self.ser = np.empty(size)
        self.ser[:] = np.nan
        self.frequency = frequency
        self.label = label

    def __len__(self):
        return len(self.ser)

    def __str__(self) -> str:
        return f"Series of length {len(self)}. Last values are: {self.ser[-10:]}"

    def append(self, x: float):
        self.ser = np.append(self.ser[1:], x)

    @property
    def movavg(self) -> np.ndarray:
        return moving_average(self.ser, size=self.frequency)

    @property
    def list(self) -> list:
        return [-1 if math.isnan(x) else x for x in self.ser.tolist()]

    def plot(self, ax: Axes):
        """Plots the line of given axes"""
        (self.line,) = ax.plot(self.ser, label=self.label)

    def update_plot(self):
        """Updates the plotted line"""
        self.line.set_ydata(self.movavg)


class PlotSeries:
    def __init__(self, list_ser: List[Series], xlim: tuple = None, ylim: tuple = None):
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
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.draw()
        self.list_ser = list_ser
        for ser in self.list_ser:
            ser.plot(self.ax)
        self.ax.grid()
        self.ax.set_xlabel("Time (frames)")
        self.ax.set_ylabel("Detection box size")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.legend()
        plt.show(block=False)

    def update(self):
        """Updates the figure"""
        for ser in self.list_ser:
            ser.update_plot()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Closes the figure"""
        plt.close(self.fig)
