import math
from typing import Callable, List

import iisignature
import numpy as np
import torch
from tqdm import tqdm

from src.testing import TestConfig
from src.testing.discriminators import Processor
from src.testing.discriminators.config import ProcessorConfig
from src.utils.auxiliary_classes.PathTransformer import PathTransformer
from src.utils.helper_functions.data_helper_functions import reweighter
from src.utils.helper_functions.test_helper_functions import get_sub_paths, get_grouped_paths
from src.utils.helper_functions.signature_helper_functions import all_words
from src.testing.experiment_functions.threshold_fitters import GammaFitter


class TruncatedAutoEvaluator(Processor):
    """
    Auto path evaluator class for Processor. Does not generate a prior, instead auto-evaluates path against itself,
    given lags.
    """

    def __init__(self, path_transformer: PathTransformer, processor_config: ProcessorConfig,
                 test_config: TestConfig):
        """
        Constructor for AutoEvaluator class.

        :param path_transformer:    PathTransformer. Object which transforms paths given config
        :param processor_config:    ProcessorConfig. Metric arguments and so on.
        :param test_config:         TestConfig. How path is split up and so on.
        """

        # Initialise config variables
        self.path_transformer = path_transformer
        self.algorithm_name   = "truncatedautoevaluator"

        self.processor_config = processor_config
        self.algorithm_kwargs = getattr(processor_config, "{}_kwargs".format(self.algorithm_name))
        self.metric_kwargs    = self.algorithm_kwargs.metric_kwargs
        self.evaluator_kwargs = self.algorithm_kwargs.evaluator_kwargs
        self.n_scores         = self.algorithm_kwargs.n_scores

        # Extract config variables
        self.n_steps          = test_config.n_steps
        self.n_paths          = test_config.n_paths
        self.offset           = test_config.offset
        self.weight_factor    = test_config.weight_factor
        self.weights          = reweighter(self.n_paths, self.weight_factor)
        self.lags             = self.evaluator_kwargs.lags

        self.signature_order     = self.metric_kwargs.signature_order
        self.scale_signature     = self.metric_kwargs.scale_signature
        self.sigma               = self.metric_kwargs.sigma
        self.similarity_function = lambda x: torch.exp(-x/(self.sigma**2))

        # Functions specific to class
        self.metric           = self.initialise_metric()
        self.path_evaluator   = self.lambda_path_evaluator

        super(TruncatedAutoEvaluator, self).__init__(
            beliefs           = np.array([]),
            path_details      = "na",
            algorithm_type    = self.algorithm_name,
            metric            = self.metric,
            prior_generator   = self.lambda_prior_generator,
            path_evaluator    = self.path_evaluator,
            path_transformer  = self.path_transformer,
            config            = self.processor_config
        )

    def initialise_metric(self) -> Callable:
        """
        Initialises the signature kernel with the given config parameters

        :return:    Callable. Signature MMD metric with specified parameters
        """

        # Wrapper function
        wrapper       = self.wrapper
        order         = self.signature_order
        scale         = self.scale_signature
        sim_func      = self.similarity_function

        def truncated_mmd(x, y):
            dim = x.shape[-1]
            _all_words = all_words(dim, order)
            if scale:
                scaler = wrapper([math.factorial(len(word)) for word in _all_words])
            else:
                scaler = wrapper(np.ones(len(_all_words)))

            sx = iisignature.sig(wrapper(x), order)
            sy = iisignature.sig(wrapper(y), order)

            scaled_sx = torch.vstack([torch.mul(scaler, s) for s in sx])
            scaled_sy = torch.vstack([torch.mul(scaler, s) for s in sy])

            kxx = torch.einsum("ip,jp->ij", scaled_sx, scaled_sx)
            kxy = torch.einsum("ip,jp->ij", scaled_sx, scaled_sy)
            kyy = torch.einsum("ip,jp->ij", scaled_sy, scaled_sy)

            return torch.mean(kxx) - 2*torch.mean(kxy) + torch.mean(kyy)

        return lambda x, y: 1 - sim_func(truncated_mmd(x, y))

    @staticmethod
    def lambda_prior_generator() -> (np.ndarray, float):
        """
        Blank. No prior to evaluate

        :return:    Dummy entries
        """
        return np.array([]), 0.0

    def lambda_path_evaluator(self, test_path: np.ndarray, **kwargs) -> List[np.ndarray]:
        """
        Path evaluator for AutoEvaluator. Returns [indexes, scores, alphas] where alphas are relative to changing
        prior distribution.

        :param test_path:       Path to evaluate
        :param kwargs:          Extra arguments (lags).
        :return:
        """
        # Init variables
        n_steps          = self.n_steps
        offset           = self.offset
        n_paths          = self.n_paths
        weights          = self.weights
        metric           = self.metric
        n_scores         = self.n_scores
        prior_dist       = []

        if kwargs == {}:
            lags = self.lags
        else:
            lags = kwargs.get("lags")

        sub_paths             = get_sub_paths(test_path, n_steps, offset)
        mmd_paths             = get_grouped_paths(sub_paths, n_paths)
        sub_paths_transformed = self.path_transformer.transform_paths(sub_paths.copy())
        mmd_paths_transformed = get_grouped_paths(sub_paths_transformed, n_paths)

        # Init scores array
        first_path = -min(lags)
        num_lags   = len(lags)
        scores     = np.zeros(mmd_paths_transformed.shape[0]-first_path)
        alphas     = np.zeros(mmd_paths_transformed.shape[0]-first_path)
        indexes    = mmd_paths[first_path:, -1, -1, 0]

        for i, paths in tqdm(enumerate(mmd_paths_transformed[first_path:])):
            this_score = 0

            for lag in lags:
                this_score += metric(paths[weights], mmd_paths_transformed[i+1+lag][weights]).item()

            final_score = this_score/num_lags
            scores[i]   = final_score

            # Handle prior stuff
            if i+1 <= n_scores:
                prior_dist.append(final_score)
                alphas[i] = 0
            else:
                prior_dist = prior_dist[1:] + [final_score]
                thresh     = GammaFitter(prior_dist).ppf(self.alpha_value)
                alphas[i]  = final_score >= thresh

        return np.c_[indexes, scores, alphas].T

    # Helper functions
    def get_final_path_length(self) -> int:
        """
        Gets the length of paths being calculated given PathTransformer arguments.

        :return:    Int. Length of paths where the MMD is being calculated over
        """
        pt_args = self.path_transformer.transformations
        n_steps = self.n_steps

        if pt_args["lead_lag_transform"][0]:
            return int(2*n_steps - 1)
        else:
            return int(n_steps)