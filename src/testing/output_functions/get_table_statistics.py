from time import time
from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score

from src.generators.Model import Model
from src.testing import TestConfig
from src.testing.discriminators import Processor
from src.testing.experiment_functions.mmd_test_functions import get_sub_paths, get_grouped_paths, generate_regime_path
from src.utils.auxiliary_classes.RegimePartitioner import RegimePartitioner
from src.utils.helper_functions.test_helper_functions import get_memberships


def get_table_statistics(processor: Processor, models: List[Model],
                         regime_partitioner: RegimePartitioner, config: TestConfig) -> np.ndarray:
    """
    Gets statistics for table.

    :param processor:               Instance of Processor class to test
    :param models:                  List of Models used to build regime path
    :param regime_partitioner:      Instance of RegimePartitioner to build regime path
    :param config:                  Named arguments for path evaluators and so on.
    :return:
    """

    # General test configs
    n_runs  = config.n_runs
    n_steps = config.n_steps
    n_paths = config.n_paths
    offset  = config.offset
    time_sim = config.time_sim
    critical_value = processor.alpha_value
    S0 = [1. for _ in range(models[0].dim)]

    # Algorithm kwargs
    algorithm_name = processor.algorithm_type
    algorithm_args = getattr(config, processor.algorithm_type + "_kwargs")
    eval_kwargs = algorithm_args.eval_kwargs
    res = np.zeros((n_runs, 5))

    for i in range(n_runs):
        regime_partitioner.generate_regime_partitions(T=time_sim, n_steps=n_steps)
        regime_changes = regime_partitioner.regime_changes
        test_path = regime_partitioner.generate_regime_change_path(models, S0)

        true_labels = np.mean(
            get_sub_paths(
                generate_regime_path(regime_changes, test_path.shape[0]),
                n_steps, offset),
            axis=1
        )

        true_labels = np.array(true_labels, dtype=np.int32)

        # Evaluate path
        init_eval = time()
        score_matrix = processor.evaluate_path(test_path, **eval_kwargs)
        eval_time = time() - init_eval

        # Get labels on sub-paths
        if algorithm_name in ["generalmmddetector", "truncatedmmddetector"]:
            sub_paths      = get_sub_paths(test_path, n_steps, offset)
            ensemble_paths = get_grouped_paths(sub_paths, n_paths)

            if n_paths > 1:
                # Get average MMD score per sub-path
                memberships = get_memberships(ensemble_paths)
                score_matrix = np.array([[np.mean(score[m]) for m in memberships] for score in score_matrix[1:, :]])
            else:
                score_matrix = score_matrix[1:, :]
            # Move to quantiles
            quantiles = processor.return_quantiles(score_matrix)
        elif algorithm_name == "anomalydetector":
            score_matrix = score_matrix[1:, :]
            quantiles = processor.return_quantiles(score_matrix)
        else:
            quantiles = []

        # Accuracy
        predictions     = np.array([[1.0*(q > critical_value) for q in quan] for quan in quantiles])
        mask1           = true_labels == 1
        mask2           = true_labels == 0

        total_accuracy     = np.max(np.array([np.mean((prediction - true_labels == 0)) for prediction in predictions]))

        regime_on_accuracy = np.max(
            np.array([np.mean((prediction[mask1] - true_labels[mask1] == 0)) for prediction in predictions])
        )

        regime_off_accuracy = np.max(
            np.array([np.mean((prediction[mask2] - true_labels[mask2] == 0)) for prediction in predictions])
        )

        # ROC AUC
        rocauc = np.max([roc_auc_score(true_labels, q) for q in quantiles])

        res[i, 0] = regime_on_accuracy
        res[i, 1] = regime_off_accuracy
        res[i, 2] = total_accuracy
        res[i, 3] = rocauc
        res[i, 4] = eval_time

    return res


def get_score_statistics(processor: Processor):
    scores = np.sort(processor.scores)
    n_priors, n_scores = scores.shape

    return scores, n_priors, n_scores
