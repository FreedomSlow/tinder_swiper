import os
import pickle

from IPython import display
import matplotlib.pyplot as plt
import torch


def read_pickle(file_path: str, not_found_return=None):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return not_found_return


class Plotter:
    def __init__(
        self,
        x_label: str = None,
        y_label: str = None,
        legend: list = None,
        x_lim: list = None,
        y_lim: list = None,
        x_scale: str = "log",
        y_scale: str = "log",
        n_rows: int = 1,
        n_cols: int = 1,
        figsize: tuple = (8, 6)
    ):

        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows * n_cols == 1:
            self.axes = [self.axes,]
        self.config_axes = lambda: self.set_axes(
            self.axes[0], x_label, y_label, x_lim, y_lim, x_scale, y_scale, legend
        )
        self.X, self.Y = None, None

    @staticmethod
    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib."""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y in zip(self.X, self.Y):
            self.axes[0].plot(x, y)
        self.config_axes()

    def plot(self):
        display.clear_output(wait=True)
        display.display(self.fig)


def try_gpu():
    if torch.cuda.device_count() > 0:
        return torch.device("cuda:0")

    return torch.device("cpu")

