# Standard imports
import math

# Matplotlib Setup
import matplotlib as mpl

mpl.use('pgf')

"""
Utility to generate PGF vector files from Python's Matplotlib plots to use in LaTeX documents. All credit goes to Nils 
Fischer. Source code repository at https://github.com/nilsleiffischer/texfig.
"""

default_width = 4  # in inches
default_ratio = 2.0 / (1 + math.sqrt(5.0))  # golden ratio

mpl.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": "xelatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "figure.figsize": [default_width, default_width * default_ratio],
    "pgf.preamble": [
        # put LaTeX preamble declarations here
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
        r"\usepackage{bm}"
        # macros defined here will be available in plots, e.g.:
        r"\newcommand{\vect}[1]{#1}",
        # You can use dummy implementations, since your  LaTeX document
        # will render these properly, anyway.
    ],
})

from matplotlib import pyplot as plt


def figure(width=default_width, ratio=default_ratio, pad=0, *args, **kwargs):
    """
    Returns a figure with an appropriate size and tight layout.

    Parameters
    ----------
    width: float
        Width of figure.
    ratio: float
        Height to width ratio.
    pad: float
        Figure padding.
    *args
    **kwargs

    Returns
    -------
    fig
    """
    fig = plt.figure(figsize=(width, width * ratio), *args, **kwargs)
    fig.set_tight_layout({
        'pad': pad
    })
    return fig


def subplots(width=default_width, ratio=default_ratio, pad=0, w_pad=None, h_pad=None, *args, **kwargs):
    """
    Returns subplots with an appropriate figure size and tight layout.

    Parameters
    ----------
    width: float
        Width of figure.
    ratio: float
        Height to width ratio.
    pad: float
        Figure padding.
    w_pad: float
        Padding (height/width) between edges of adjacent subplots, as a fraction of the font size. Defaults to pad.
    h_pad: float
        Padding (height/width) between edges of adjacent subplots, as a fraction of the font size. Defaults to pad.
    *args
    **kwargs

    Returns
    -------
    fig, axes
    """
    fig, axes = plt.subplots(figsize=(width, width * ratio), *args, **kwargs)
    fig.set_tight_layout({
        'pad': pad,
        'w_pad': w_pad,
        'h_pad': h_pad
    })
    return fig, axes


def savefig(filename, *args, **kwargs):
    """
    Save both a PDF and a PGF file with the given filename.

    Parameters
    ----------
    filename:str
        Filename of plots.
    *args
    **kwargs
    """
    plt.savefig(filename + '.pdf', *args, **kwargs)
    plt.savefig(filename + '.pgf', *args, **kwargs)
