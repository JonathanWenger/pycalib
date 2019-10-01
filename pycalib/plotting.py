# Standard imports
import numpy as np
import scipy.stats
import pandas as pd

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# scikit-learn
import sklearn.metrics

# Package imports
import pycalib.scoring
import pycalib.texfig as texfig


def reliability_diagram(y, p_pred, filename, title="Reliability Diagram", n_bins=100, show_ece=False, show_legend=False,
                        model_name = None, xlim=None, plot_height=4, plot_width=4):
    """
    Plot a reliability diagram

    This function plots the reliability diagram [1]_ [2]_ and histograms from the given confidence estimates. Reliability diagrams
    are a visual aid to determine whether a classifier is calibrated or not.

    Parameters
    ----------
    y : array, shape = [n_methods, n_samples]
        Ground truth labels.
    y_pred : array, shape = [n_methods, n_samples]
        Predicted labels.
    p_pred : array or list
        Array of confidence estimates. If this is a list, multiple reliability diagrams will be plotted and arranged
        side-by-side.
    filename : str
        Path or name of output plot files.
    title : str or list
        Title of plot. If `p_pred` is a list, titles of each individual reliability diagram.
    n_bins : int, optional, default=20
        The number of bins into which the `y_pred` are partitioned.
    show_ece : bool
        Whether the expected calibration error (ECE) should be displayed in the plot.
    show_legend : bool
        Whether the legend should be displayed in the plot.
    model_name : str
        Name of the model from which the probabilities were generated. Displayed when showing the legend.
    xlim : array, shape = (2,), default=None
        X-axis limits. If note provided inferred from y.

    References
    ----------
    .. [1] DeGroot, M. H. & Fienberg, S. E. The Comparison and Evaluation of Forecasters. Journal of the Royal
           Statistical Society. Series D (The Statistician) 32, 12–22.
    .. [2] Niculescu-Mizil, A. & Caruana, R. Predicting good probabilities with supervised learning in Proceedings of
           the 22nd International Conference on Machine Learning (2005)

    """
    # Initialization
    if xlim is None:
        n_classes = len(np.unique(y))
        xlim = [1 / n_classes, 1]

    # Check whether multiple reliability diagrams should be plotted
    n_plots = 1
    if isinstance(p_pred, list):
        n_plots = len(p_pred)

    # Define bins
    bins = np.linspace(xlim[0], xlim[1], n_bins + 1)

    # Plot reliability diagram
    fig, axes = texfig.subplots(nrows=2, ncols=n_plots, width=plot_width, ratio=plot_height/plot_width,
                                sharex=True, sharey=True,
                                gridspec_kw={'height_ratios': [2, 1]})

    for i in range(n_plots):
        if n_plots > 1:
            current_plot_axes = axes[:, i]
            y_pred = np.argmax(p_pred[i], axis=1)
            p_max = np.max(p_pred[i], axis=1)
            p_pred_current = p_pred[i]
        else:
            current_plot_axes = axes
            y_pred = np.argmax(p_pred, axis=1)
            p_max = np.max(p_pred, axis=1)
            p_pred_current = p_pred

        # Calibration line
        current_plot_axes[0].plot(xlim, xlim, linestyle='--', color='grey')

        # Compute bin means and empirical accuracy
        bin_means = np.linspace(xlim[0] + xlim[1] / (2 * n_bins), xlim[1] - xlim[1] / (2 * n_bins), n_bins)
        empirical_acc = scipy.stats.binned_statistic(p_max,
                                                     np.equal(y_pred, y).astype(int),
                                                     bins=n_bins,
                                                     range=xlim)[0]
        empirical_acc[np.isnan(empirical_acc)] = bin_means[np.isnan(empirical_acc)]

        # Plot accuracy
        current_plot_axes[0].step(bins, np.concatenate(([xlim[0]], empirical_acc)), '-', label=model_name)
        x = np.linspace(xlim[0], xlim[1], 1000)
        bin_ind = np.digitize(x, bins)[1:-1] - 1
        current_plot_axes[0].fill_between(x[1:-1], x[1:-1], (bin_means + (empirical_acc - bin_means))[bin_ind], facecolor='k', alpha=0.2)
        if i == 0:
            current_plot_axes[0].set_ylabel('accuracy')
        if show_legend:
            current_plot_axes[0].legend(loc='upper left')
        if title is not None and n_plots > 1:
            current_plot_axes[0].set_title(title[i])
        elif title is not None:
            current_plot_axes[0].set_title(title)

        # Plot histogram
        hist, ex = np.histogram(p_max, bins=bins)
        current_plot_axes[1].fill_between(bins, np.concatenate(([0], hist / np.sum(hist))), lw=0.0, step="pre")
        current_plot_axes[1].set_xlabel('confidence $\\hat{z}$')
        if i == 0:
            current_plot_axes[1].set_ylabel('sample frac.')
        current_plot_axes[1].set_xlim(xlim)
        fig.align_labels()

        # Add textbox with ECE
        if show_ece:
            ece = pycalib.scoring.expected_calibration_error(y=y, p_pred=p_pred_current, n_bins=n_bins)
            anchored_text = AnchoredText("$\\textup{ECE}_1 = " + "{:.3f}$".format(ece), loc='lower right')
            anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            anchored_text.patch.set_edgecolor("0.8")
            anchored_text.patch.set_alpha(0.9)
            current_plot_axes[0].add_artist(anchored_text)

    # Save to file
    texfig.savefig(filename=filename, bbox_inches='tight', pad_inches=0)


# def confidence_diagram(y, y_pred, p_pred, n_classes=None, file=None, plot=True, n_bins=20, color=None, **kwargs):
#     """
#     Plot a confidence diagram
#
#     Plots a diagram showing both distributions of over- and underconfidence.
#
#     Parameters
#     ----------
#     y : array, shape = [n_samples]
#         Ground truth labels.
#     y_pred : array, shape = [n_samples]
#         Predicted labels.
#     p_pred : array, shape = [n_samples, n_classes]
#         Array of confidence estimates.
#     file : str, optional
#         File name of output plot.
#     plot : bool
#         Should the generated plot be shown?
#     n_bins: int, optional, default=20
#         Number of bins to use for the histograms.
#     """
#
#     # Determine number of classes
#     if n_classes is None:
#         n_classes = len(np.unique(y))
#
#     # Find prediction confidence
#     p_max = np.max(p_pred, axis=1)
#
#     # Define bins
#     bins = np.linspace(0, 1, n_bins + 1)
#
#     # Plot overconfidence
#     plt.subplot(2, 1, 1)
#     plt.hist(p_max[y != y_pred], range=[1 / n_classes, 1], bins=bins, color=color, **kwargs)
#     plt.axvline(pycalib.scoring.overconfidence(y, y_pred, p_pred), linewidth=3, color='k')
#     plt.xlim([1 / n_classes, 1])
#     plt.ylabel('Count')
#     plt.title('Confidence of false predictions')
#
#     # Plot 1 - underconfidence
#     plt.subplot(2, 1, 2)
#     plt.xlim([1 / n_classes, 1])
#     plt.hist(p_max[y == y_pred], range=[1 / n_classes, 1], bins=bins, color=color, **kwargs)
#     plt.axvline(1 - pycalib.scoring.underconfidence(y, y_pred, p_pred), linewidth=3, color='k')
#     plt.xlabel('Predicted Probability')
#     plt.ylabel('Count')
#     plt.title('Confidence of correct predictions')
#
#     # Save plot
#     plt.subplots_adjust(hspace=0.5)
#     if file is not None:
#         plt.savefig(file)
#     if plot:
#         plt.show()
#
#     return plt
