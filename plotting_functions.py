import matplotlib.pyplot as plt
import numpy as np
import os
import string


def set_ax_labels(ax: plt.Axes, title: str, x: str, y: str):
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def set_labels(axes: [plt.Axes, list, np.array],
               titles: [str, list, np.array],
               x: [str, list, np.array] = "x",
               y: [str, list, np.array] = "y") -> None:
    iterable_title = False
    iterable_x = False
    iterable_y = False
    try:
        iter(axes)
        iterable_ax = True
        try:
            iter(titles)
            if isinstance(titles, str):
                raise TypeError
            iterable_title = True
        except TypeError:
            iterable_title = False
        try:
            iter(x)
            if isinstance(x, str):
                raise TypeError
            iterable_x = True
        except TypeError:
            iterable_x = False
        try:
            iter(y)
            if isinstance(y, str):
                raise TypeError
            iterable_y = True
        except TypeError:
            iterable_y = False
    except TypeError:
        iterable_ax = True
    if iterable_ax:
        for i, ax in enumerate(axes):
            set_ax_labels(ax, titles[i] if iterable_title else titles, x[i] if iterable_x else x,
                          y[i] if iterable_y else y)
    else:
        set_ax_labels(axes, titles, x, y)


def savefig(fig: plt.Figure, title: str) -> None:
    if title[-4:].lower() != ".png":
        title += ".png"
    fig.tight_layout()
    for i, ax in enumerate(fig.axes):
        ax.text(-0.05, 1.01, string.ascii_uppercase[i], transform=ax.transAxes,
                size=20, weight='bold')
#    fig.savefig(os.path.join(FIG_DIR, title))
