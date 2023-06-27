from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.helper_functions.plot_helper_functions import golden_dimensions, make_grid
from src.utils.helper_functions.data_helper_functions import get_log_returns
from src.utils.helper_functions.global_helper_functions import get_project_root


def plot_path_experiment_result(path: np.ndarray, test_score: np.ndarray, path_splits: List[int],
                                diff=False, one_dim=True, save_image=(False, ""), y_label="$MMD$", zero_bnd=False) -> None:
    """
    Plots a regime-changed path against the score from a ProcessTest instance.

    :param path:            Path that the test was conducted over
    :param test_score:      Array of test scores with attached index for plotting against path
    :param path_splits:     Where regime changes occurred
    :param diff:            Boolean. If set to TRUE, returns log-returns (differenced) plot.
    :param one_dim:         Boolean. Only plot the first dimension.
    :param save_image:      Boolean/string. Whether to save the image and what the filename should be.
    :param y_label          Metric axis label
    :param zero_bnd:        Plot horizontal line at 0 or not
    :return:                Plot of regime-changed path and its associated score
    """

    plt.figure(figsize=golden_dimensions(10))

    dim = int(path.shape[-1]  - 1)

    # Plot helpers
    changes = [0] + path_splits + [int(path.shape[0])]
    colors = ["seagreen", "tomato"]
    mmd_colors = sns.color_palette("Greys", n_colors=int(test_score.shape[0]-1))
    labels = [r"$\mathcal{M}_1$", r"$\mathcal{M}_2$"]
    indexes = test_score[0, :]

    plot_path = path.copy()
    plot_dim = 1 if one_dim else dim

    # Plot regime changes of path
    for di in range(plot_dim):
        plt.subplot(plot_dim, 1, di+1)
        ax1 = plt.gca()
        ax1.set_xlabel("$t$")
        ax1.set_ylabel("$S$")

        if diff:
            plot_path[:, 1+di] = get_log_returns(plot_path[:, 1+di])

        for i, (start, end) in enumerate(zip(changes[:-1], changes[1:])):

            ind = int(i % 2)

            plt.plot(
                plot_path[start:end, 0],
                plot_path[start:end, 1+di],
                color=colors[ind],
                alpha=0.75,
                label=labels[ind]
            )

        make_grid()
        ax2 = ax1.twinx()
        ax2.set_ylabel(y_label)

        for k, score in enumerate(test_score[1:]):
            ax2.plot(
                indexes,
                score,
                color=mmd_colors[k],
                alpha=0.25,
                label="{}_score_{}".format(y_label, k+1)
            )

        if zero_bnd:
            ax2.axhline(0, color="black", alpha=0.15, linestyle="dashed")
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), prop={'size': 12}, loc="upper left")
        ax2.legend(loc="upper right", prop={'size': 12})

    if save_image[0]:
        plt.savefig(get_project_root().as_posix() + "/data/images/mmd_scores/{}.png".format(save_image[1]), dpi=300)

    plt.show()


def plot_path_test_threshold(sub_paths: np.ndarray, alphas: np.ndarray, path_splits: List[float], one_dim=False,
                             as_timestamp=False, save_image=(False, ""), title="") -> None:
    """
    Plots a shaded path, where the intensity of the shading is given by an alphas vector.

    :param sub_paths:       N x n_paths x d array of sub-paths for plotting
    :param alphas:          List of alpha scores for each sub-path
    :param path_splits:     List of regime change times as float
    :param one_dim:         Whether to plot one dimension or more
    :param as_timestamp:    Boolean, whether to make indexes timestamps or not
    :param save_image:      Boolean, string. Whether to save the image or not, with associated name.
    :param title:           Custom title to use for plot
    :return:                
    """

    plt.figure(figsize=golden_dimensions(10))
    dim = sub_paths.shape[-1] - 1
    plot_dim = dim if not one_dim else 1

    splits = [[x, y] for x, y in zip(path_splits[::2], path_splits[1::2])]

    for di in range(plot_dim):
        plt.subplot(plot_dim, 1, 1 + di)
        ax1 = plt.gca()
        ax1.set_xlabel("$t$")
        ax1.set_ylabel("$S$")

        for p, alph in zip(sub_paths, alphas):
            if alph == 0.0:
                plt.plot(p[:, 0], p[:, 1 + di], color='dodgerblue', linestyle="dashed", alpha=0.05)
            else:
                plt.plot(p[:, 0], p[:, 1 + di], color='dodgerblue', alpha=alph)

        for split in splits:
            plt.axvspan(split[0], split[1], color='tomato', alpha=0.2, label="regime_change")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop={'size': 12}, loc="upper left")
        make_grid()

        if as_timestamp:
            indexes = sub_paths[:, :, 0].flatten()
            to = int(0.10*indexes.shape[0])
            ax1.set_xticks(indexes[to::2*to])
            ax1.set_xticklabels(list(map(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'), indexes[to::2*to])))

    plt.title(title)
    if save_image[0]:
        plt.savefig(get_project_root().as_posix() + "/data/images/alpha_plots/{}.png".format(save_image[1]), dpi=300)

    plt.show()
