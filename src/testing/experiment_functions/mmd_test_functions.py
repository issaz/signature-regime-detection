import os
from typing import List, Callable

import numpy as np
from tqdm import tqdm

from src.generators.Model import Model
from src.generators.config import ModelConfig
from src.testing import TestConfig
from src.testing.discriminators import Processor
from src.utils.auxiliary_classes.RegimePartitioner import RegimePartitioner
from src.utils.helper_functions.global_helper_functions import get_project_root
from src.utils.helper_functions.test_helper_functions import get_alphas, get_memberships
from src.utils.helper_functions.test_helper_functions import get_sub_paths, get_grouped_paths


def generate_deterministic_test_path(model_on: Model, model_off: Model, T: float, S0: List[float]) -> np.ndarray:
    """
    Generates a test path which switches regime at a deterministic time (the middle). Each regime lasts for T years.

    :param model_on:        Model representing standard regime
    :param model_off:       Model representing regime change
    :param T:               Time (years) for each path to elapse
    :param S0:              List[float] of initial stock prices
    :return:                Concatenated test path
    """

    r1_path = model_on.sim_path(T=T, S0=S0)
    final_stock_values = r1_path[-1, 1:]
    r2_path = model_off.sim_path(T=T, S0=S0, time_add=r1_path[-1, 0])

    for i, s in enumerate(final_stock_values):
        r2_path[:, i + 1] *= s

    return np.vstack([r1_path, r2_path])


def get_set_paths(models: List[Model], time: float, path_bank_size: int, config: ModelConfig,
                  overwrite=False) -> np.ndarray:
    """
    Gets a bank of paths corresponding to a List of models with associated parameters of length time (years).

    :param models:              Models to get path banks of
    :param time:                Amount of time to sim each path for (years)
    :param path_bank_size:      Size of path bank
    :param config:              Instance of ModelConfig to determine year mesh
    :param overwrite:           Boolean. Whether to overwrite the current bank of paths
    :return:                    NP array of bank paths corresponding to each Model
    """

    assert len(set([model.dim for model in models])) == 1, "ERROR: Models have different dimensions"

    _dim = models[0].path_dim

    # Establish base model NOTE: Only handle models from the same family for now.
    init_stocks = [[1. for _ in range(model.dim)] for model in models]

    n_steps  = int(time*config.year_mesh) + 1
    path_data = np.zeros((len(models), path_bank_size, n_steps, _dim + 1))

    for i, model in enumerate(models):
        model_type, params = model.model_type, model.params

        these_params = params[0]  # Again we assume all params are the same for now

        params_str = '_'.join([str(p).replace(".", "") for p in these_params])

        this_path_args = '/data/paths/{}_{}_d_{}_years_{}_mesh_{}_vol_{}.npy'.format(
            model_type,
            params_str,
            model.dim,
            str(np.round(time, 4)).replace(".", ""),
            config.year_mesh,
            config.attach_volatility
        )

        data_path = get_project_root().as_posix() + this_path_args

        if os.path.exists(data_path) and not overwrite:
            path_data[i] = np.load(data_path, allow_pickle=True)
        else:
            sim_path = model.sim_path
            S0 = init_stocks[i]

            generated_paths = np.array([
                    sim_path(T=time, S0=S0) for _ in tqdm(range(path_bank_size), position=0, leave=True)
            ])

            # Save paths and output for function
            np.save(data_path, generated_paths, allow_pickle=True)
            path_data[i] = generated_paths

    return np.array(path_data)


def generate_regime_path(regime_changes: np.ndarray, path_length: int) -> np.ndarray:
    """
    Generates path with ones indicating where a regime change occurred, and zeros otherwise.

    :param regime_changes:  Array of 2-lists, starts and ends of regimes in time units
    :param path_length:     Overall length of path
    :return:                Regime change path
    """

    regime_path = np.zeros(path_length)

    for ind in regime_changes:
        regime_path[ind[0]:ind[1]] = 1

    return regime_path


def score_mmd_test_synthetic_data(regime_changes: np.ndarray, path_length: int, year_mesh: int, critical_value: float,
                                  scores_array: np.ndarray) -> dict:
    """
    Gives an accuracy score related to a given run of the MMD classifier.

    :param regime_changes:  When the regime changes started and ended
    :param path_length:     Length of test path (in grid_point units)
    :param year_mesh:       Year mesh from ModelConfig
    :param critical_value:  Critical value associated to instance of Processor
    :param scores_array:    MMD scores corresponding to ensemble paths
    :return:                Dictionary of accuracy scores for regime_off, regime_on, and in total. Scores are given by
                            1 - E[|y - y_hat|].
    """

    regime_path = generate_regime_path(regime_changes, path_length)

    tests_passed = scores_array[np.where(scores_array[:, 1] >= critical_value)[0], 0]
    tests_failed = scores_array[np.where(scores_array[:, 1]  < critical_value)[0], 0]

    tp_ind = np.array([int(t * year_mesh) for t in tests_passed])
    tf_ind = np.array([int(t * year_mesh) for t in tests_failed])

    abs_on_scores  = np.abs(regime_path[tp_ind] - 1)
    abs_off_scores = np.abs(regime_path[tf_ind])
    concat_scores  = np.concatenate([abs_on_scores, abs_off_scores])

    score_dict = {
        "regime_on_accuracy" : 1-np.mean(abs_on_scores),
        "regime_off_accuracy": 1-np.mean(abs_off_scores),
        "total_accuracy":      1-np.mean(concat_scores)
    }

    return score_dict


def alpha_score_function(regime_changes: np.ndarray, path_length: int, memberships: list,
                         test_alphas: np.ndarray, test_data: list) -> dict:
    """
    Gives the score of a given MMD clustering algorithm, given alphas and correct regime classifications

    :param regime_changes:      List of 2-lists of regime change locations
    :param path_length:         Length of synthetic path
    :param memberships:         Memberships of sub-paths of test path, ragged list of lists
    :param test_alphas:         Alpha scores from MMD run
    :param test_data:           List of [n_steps, offset, normalise, n_paths] values
    :return:                    Dictionary of regime-on, regime-off and total accuracy
    """

    # Initialise data
    n_steps, offset, n_paths = test_data

    # Generate regime path
    regime_path = generate_regime_path(regime_changes, path_length)

    # Extract sub paths and build true alphas path
    true_sub_paths = get_sub_paths(regime_path, n_steps, offset)
    true_grouped_paths = get_grouped_paths(true_sub_paths, n_paths)
    pct_true_passed = np.sum(true_grouped_paths, axis=1) / n_paths
    pct_true_passed = pct_true_passed[:, 0]

    true_alphas = []
    for m in memberships:
        true_alphas.append(np.mean(pct_true_passed[m]))

    # Compare test and true alphas
    alphas = np.array(test_alphas)
    true_alphas = np.array(true_alphas)

    regime_on_mask = np.where(true_alphas > 0.)[0]
    regime_off_mask = np.where(true_alphas == 0)[0]

    # Calculate accuracy scores
    regime_on_accuracy = 1 - np.mean(np.abs(alphas[regime_on_mask] - true_alphas[regime_on_mask]))
    regime_off_accuracy = 1 - np.mean(np.abs(alphas[regime_off_mask] - true_alphas[regime_off_mask]))
    total_accuracy = 1 - np.mean(np.abs(alphas - true_alphas))

    final_scores = {
        "regime_on": regime_on_accuracy,
        "regime_off": regime_off_accuracy,
        "total": total_accuracy
    }

    return final_scores


def n_alpha_test(tests: int, detector: Processor, regime_partitioner: RegimePartitioner, test_config: TestConfig,
                 regime_partitioner_args: dict, model_pairs: List[Model]) -> dict:
    """
    TODO

    :param tests:
    :param detector:
    :param regime_partitioner:
    :param test_config:
    :param regime_partitioner_args:
    :param model_pairs:

    :return:
    """

    regime_on_scores = 0
    regime_off_scores = 0
    total_scores = 0

    n_paths = test_config.n_paths
    n_steps = test_config.n_steps
    offset  = test_config.offset
    dim     = list(set((m.dim for m in model_pairs)))[0]
    res = {}

    S0 = [1. for _ in range(dim)]

    for _ in range(tests):
        regime_partitioner.generate_regime_partitions(**regime_partitioner_args)
        test_path = regime_partitioner.generate_regime_change_path(model_pairs, S0)

        sub_paths = get_sub_paths(test_path, n_steps, offset)
        mmd_paths = get_grouped_paths(sub_paths, n_paths)

        scores_array = detector.evaluate_path(test_path, evaluation="total")
        memberships = get_memberships(mmd_paths)
        mmd_alphas = get_alphas(memberships, scores_array[:, 1], detector.critical_value)

        scores_dict = alpha_score_function(
            regime_changes = regime_partitioner.regime_changes,
            path_length    = len(test_path),
            memberships    = memberships,
            test_alphas    = mmd_alphas,
            test_data      = [n_steps, offset, n_paths]
        )

        regime_on_scores += scores_dict.get("regime_on")
        regime_off_scores += scores_dict.get("regime_off")
        total_scores      += scores_dict.get("total")

    res["regime_on"]  = regime_on_scores/tests
    res["regime_off"] = regime_off_scores/tests
    res["total"]      = total_scores/tests

    return res


def generate_mmd_histogram(beliefs: np.ndarray, metric: Callable, n_samples: int, n_paths: int) -> np.ndarray:
    """
    Generates a histogram of MMD scores under a given metric and beliefs

    :param beliefs:     Tensor of belief paths
    :param metric:      Metric to be used
    :param n_samples:   Number of atoms
    :param n_paths:     Number of paths to sim
    :return:            Histogram scores
    """

    # Generate values
    _, path_bank_size, _, _ = beliefs.shape

    rand_ints = np.random.randint(0, path_bank_size, size=(n_samples, 2, n_paths))
    scores = np.zeros(n_samples)

    for i, randi in tqdm(enumerate(rand_ints), position=0):
        x = beliefs[randi[0]]
        y = beliefs[randi[1]]

        scores[i] = metric(x, y)

    return scores


def get_beliefs_from_config(test_config: TestConfig, model_config: ModelConfig, overwrite=False):
    """
    Returns beliefs, belief details, and model pairs from TestConfig file

    :param test_config:         Instance of TestConfig
    :param model_config:        Instance of ModelConfig
    :param overwrite:           Whether to overwrite beliefs stored on disk
    :return:                    Beliefs, belief details, and model pairs
    """

    # Get configurations
    path_bank_size = test_config.path_bank_size
    n_steps        = test_config.n_steps
    time           = (n_steps-1)/model_config.year_mesh

    # Get models from config file
    beliefs_name, beliefs_params = test_config.belief_models, test_config.belief_params
    model_pair_names, model_pair_params = test_config.model_pair_names, test_config.model_pair_params

    # Instantiate model classes
    belief_models = [Model(name, theta, model_config) for name, theta in zip(beliefs_name, beliefs_params)]
    model_pairs = [Model(name, theta, model_config) for name, theta in zip(model_pair_names, model_pair_params)]

    # Generate beliefs
    beliefs = get_set_paths(
        belief_models,
        time,
        path_bank_size,
        model_config,
        overwrite
    )

    belief_details = [
        name + "_" + "_".join([str(p).replace(".", "") for p in params[0]])
        for name, params in zip(beliefs_name, beliefs_params)
    ]

    return beliefs, belief_details, model_pairs
