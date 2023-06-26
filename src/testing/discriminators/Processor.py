from typing import Callable

import torch
import numpy as np
from tqdm import tqdm

from src.utils.auxiliary_classes.PathTransformer import PathTransformer
from src.testing.discriminators.config import ProcessorConfig


class Processor(object):
    """
    Generic class for object that process/categorizes a path, given a prior (path bank).
    """

    def __init__(self, beliefs: np.ndarray, algorithm_type: str, path_details: str, metric: Callable,
                 prior_generator: Callable, path_evaluator: Callable, path_transformer: PathTransformer,
                 config: ProcessorConfig):
        """
        Instantiates a general instance of a Processor, which seeks to identify paths from an anomalous grouping under
        a prior (empirical) distribution, given by the path bank

        :param beliefs:             k x n x l array. Bank of paths to build prior over
        :param algorithm_type:      String. Name of algorithm (for accessing specific configuration values)
        :param path_details:        String. Details of the prior path bank (name of process, parameters)
        :param metric:              Callable. Metric used on the space of paths
        :param prior_generator:     Callable. Function used to generate prior. Must include metric as an argument
        :param path_evaluator:      Callable. Function that returns score vector along a given path
        :param path_transformer:    PathTransformer. Object that transforms paths given a configuration file
        :param config:              Config. Configuration file for Processor class and the specific instance of the
                                    algorithm as well
        """

        # Define class variables
        self.beliefs          = beliefs
        self.algorithm_type   = algorithm_type
        self.path_details     = path_details
        self.metric           = metric
        self.prior_generator  = prior_generator
        self.path_evaluator   = path_evaluator
        self.path_transformer = path_transformer
        self.algorithm_kwargs = getattr(config, algorithm_type + "_kwargs")
        self.alpha_value      = config.alpha_value
        self.device           = config.device
        self.jobs             = config.jobs
        self.overwrite_prior  = config.overwrite_prior

        # Build prior
        self.scores, self.critical_value = self.build_prior()

    def transform_path_bank(self):
        """
        Transforms bank of paths via PathTransformer object

        :return:    Array. Transformed bank of paths
        """

        beliefs = self.beliefs
        if len(beliefs.shape) != 1:
            transformed_beliefs = np.zeros(shape=beliefs.shape)

            for i, belief in enumerate(beliefs.copy()):
                transformed_beliefs[i] = self.path_transformer.transform_paths(belief)
        else:
            # Beliefs have an irregular number of paths in them
            transformed_beliefs = []
            for i, belief in enumerate(beliefs.copy()):
                transformed_beliefs.append(self.path_transformer.transform_paths(belief))

        return transformed_beliefs

    def build_prior(self) -> (np.ndarray, float):
        """
        Returns bank of scores associated to a path bank to build prior distribution. Also returns the critical value
        of the distribution.

        :return: Vector of scores and a float defining the alpha-critical value of the prior distribution
        """

        return self.prior_generator()

    def evaluate_path(self, test_path: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates an entire path under the metric of the Processor class.

        :param test_path:       Array. Bank of paths to evaluate. Should be of form N x M x d where N is the number of
                                paths, M the length of each path, and d the dimension
        :param kwargs:          Dict of named arguments to pass to individual instance of class
        :return:                Array of scores under metric of Processor class
        """

        assert (len(test_path.shape) == 2), "Paths need to be of shape  M x d, length M, dimension d"

        return self.path_evaluator(test_path, **kwargs)

    def _generate_distance_matrix(self, path_object: np.ndarray) -> np.ndarray:
        """
        Private function to generate distance matrix depending on path object

        :param path_object:     Collection of path(s) to generate object from
        :return:                Distance matrix D
        """

        num_paths = path_object.shape[0]
        res = np.zeros((num_paths, num_paths))

        for i in tqdm(range(num_paths)):
            path_i = path_object[i]
            for j in range(i, num_paths):
                path_j = path_object[j]
                score = self.metric(path_i, path_j)
                res[i, j] = score
                if i != j:
                    res[j, i] = score

        return res

    def return_quantiles(self, score_matrix: np.ndarray) -> np.ndarray:
        """
        Returns associated quantiles to a score vector

        :param score_matrix:    Vector of MMD scores
        :return:                Associated quantiles to scores
        """
        assert self.path_details != "na", "Cannot provide quantiles with no prior scores"

        ranked_scores = np.sort(self.scores.copy())
        n_priors, n_scores = ranked_scores.shape

        quantiles = np.array([
            [np.argmin(np.abs(ranked_scores[i] - m)) / n_scores for m in score_matrix[i]]
            for i in range(n_priors)
        ])

        return quantiles

    # Helper functions
    def wrapper(self, paths: iter) -> torch.tensor:
        """
        Wraps a numpy array around a PyTorch tensor, to compute the MMD using the SigKernel package.

        :param paths:   Array. Path to be wrapped.
        :return:        Torch tensor initialised on the GPU
        """

        return torch.tensor(paths, dtype=torch.float64, device=self.device)
