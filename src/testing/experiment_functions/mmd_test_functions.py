import os
from typing import List

import numpy as np
from tqdm import tqdm

from src.generators.Model import Model
from src.generators.config import ModelConfig
from src.testing import TestConfig
from src.utils.helper_functions.global_helper_functions import get_project_root, mkdir
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
        path_exists = os.path.exists(data_path)

        if path_exists and not overwrite:
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
    true_alphas    = true_sub_paths.mean(axis=1)

    regime_on_mask = true_alphas > 0.
    regime_off_mask = true_alphas == 0.

    # Calculate accuracy scores
    regime_on_accuracy = 1 - np.mean(np.abs(test_alphas[regime_on_mask] - true_alphas[regime_on_mask]))
    regime_off_accuracy = 1 - np.mean(np.abs(test_alphas[regime_off_mask] - true_alphas[regime_off_mask]))
    total_accuracy = 1 - np.mean(np.abs(test_alphas - true_alphas))

    final_scores = {
        "regime_on": regime_on_accuracy,
        "regime_off": regime_off_accuracy,
        "total": total_accuracy
    }

    return final_scores


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
