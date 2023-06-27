import matplotlib.pyplot as plt
import numpy as np

from src.testing.discriminators import Processor
from src.utils.helper_functions.global_helper_functions import get_project_root
from src.utils.helper_functions.data_helper_functions import get_log_returns

GOLDEN_RATIO = (1+np.sqrt(5))/2


def golden_dimensions(width: float, reverse=True) -> tuple:
    """
    Returns a tuple of l x w in the golden ratio

    :param width:   Width parameter
    :param reverse: Whether to make the plot wider (TRUE) or taller (FALSE).
    :return:        Tuple of l x w in golden ratio
    """

    if reverse:
        return GOLDEN_RATIO*width, width
    else:
        return width, GOLDEN_RATIO*width


def make_grid(axis=None):
    _plt_obj = axis if axis is not None else plt
    getattr(_plt_obj, "grid")(visible=True, color='grey', linestyle=':', linewidth=1.0, alpha=0.3)
    getattr(_plt_obj, "minorticks_on")()
    getattr(_plt_obj, "grid")(visible=True, which='minor', color='grey', linestyle=':', linewidth=1.0, alpha=0.1)


def plot_beliefs(processor: Processor, width=5, n_samples=64, reverse=False, transformed=False) -> None:
    """
    Plots a histogram corresponding to Processor scores

    :param processor:           Instance of Processor class
    :param width:               Width of each histogram plot image (height determined from width)
    :param reverse:             Whether plot should be taller (TRUE) or wider (FALSE)
    :param transformed:         Whether to apply path transformer object to paths or not
    :param n_samples:           Number of path plot samples
    :return:                    None
    """

    p, _, _, d = processor.beliefs.shape
    beliefs = processor.beliefs
    algorithm_type = processor.algorithm_type

    state_dim = int(d - 1)

    if reverse:
        _figsize = (state_dim * width, p * width * GOLDEN_RATIO)
    else:
        _figsize = (state_dim * width * GOLDEN_RATIO, p * width)

    fig, axes = plt.subplots(p, state_dim, figsize=_figsize)

    ts = beliefs[0, 0, :, 0]

    if p == 1:
        axes = np.array([axes])

    if state_dim == 1:
        axes = np.array([[ax] for ax in axes])

    for i, ax in enumerate(axes):
        if transformed:
            path_bank = processor.path_transformer.transform_paths(beliefs[i])
        else:
            path_bank = beliefs[i]

        subsample_plot_beliefs = path_bank[:n_samples, :, 1:]

        for j in range(state_dim):
            these_plot_beliefs = subsample_plot_beliefs[..., j]
            plot_flag = True

            for path in these_plot_beliefs:
                ax[j].plot(ts, path, alpha=0.5, color='dodgerblue', label=f"dim_{j + 1}" if plot_flag else "")
                plot_flag = False

            make_grid(axis=ax[j])
            ax[j].legend()
            ax[j].set_xlabel("$t$")
            ax[j].set_title(f"Distribution of beliefs for: {algorithm_type}, belief {i + 1}, dim {j + 1}")
    plt.show()


def plot_scores(processor: Processor, width=5, reverse=False) -> None:
    """
    Plots a histogram corresponding to Processor scores

    :param processor:           Instance of Processor class
    :param width:               Width of each histogram plot image (height determined from width)
    :param reverse:             Whether plot should be taller (TRUE) or wider (FALSE)
    :return:                    None
    """

    p, N           = processor.scores.shape
    scores         = processor.scores
    critical_value = processor.critical_value
    algorithm_type = processor.algorithm_type

    _scoring_flag = critical_value.shape[-1] == 2

    if reverse:
        _figsize = (width, p * width * GOLDEN_RATIO)
    else:
        _figsize = (width * GOLDEN_RATIO, p * width)

    fig, axes = plt.subplots(p, 1, figsize=_figsize)

    if p == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.hist(scores[i], bins=int(N/10), alpha=0.6, color='dodgerblue', density=True, label="scores")
        ax.axvline(critical_value[i], color="tomato", alpha=0.8, label="$c_a$")
        make_grid(axis=ax)
        ax.legend()
        ax.set_xlabel("score_values")
        ax.set_title(f"Distribution of scores for: {algorithm_type}, belief {i+1}")
    plt.show()


def plot_regime_change_path(path: np.ndarray, regime_changes: np.ndarray, log_returns=False, one_dim=False):
    """
    Plots path with regime change periods highlighted.

    :param path:            Path to plot regime changes over
    :param regime_changes:  Array of Array with [start, end] regime changes normalised for time
    :param log_returns:     Bool. Whether the path is of prices (FALSE) or log-returns (TRUE)
    :param one_dim:         Bool. Whether to only plot the first dimension or not.
    :return:                None. Displays required plot
    """

    path_dim = path.shape[-1] - 1 if not one_dim else 1
    plt.figure(figsize=golden_dimensions(10))

    def wrapper(x: np.ndarray) -> np.ndarray:
        if log_returns:
            return get_log_returns(x)
        else:
            return x

    for d in range(path_dim):
        plt.subplot(path_dim, 1, d+1)
        plt.plot(
            path[:, 0],
            wrapper(path[:, d+1]),
            color='black',
            label='{}_{}'.format('log_ret' if log_returns else 'price_path', d+1)
        )

        for c in regime_changes:
            plt.axvspan(c[0], c[1], color='red', alpha=0.25, label='regime_switch')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop={'size': 12})

        make_grid()
        plt.tight_layout()

    plt.show()
