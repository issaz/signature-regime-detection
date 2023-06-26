from typing import Callable, List

import numpy as np
from higherOrderKME import sigkernel
from tqdm import tqdm
import torch

from src.testing import TestConfig
from src.testing.discriminators import Processor
from src.testing.discriminators.config import ProcessorConfig
from src.utils.auxiliary_classes.PathTransformer import PathTransformer
from src.utils.helper_functions.data_helper_functions import reweighter
from src.utils.helper_functions.test_helper_functions import get_sub_paths, get_grouped_paths
from src.testing.experiment_functions.threshold_fitters import GammaFitter


class AutoEvaluator(Processor):
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
        self.algorithm_name   = "autoevaluator"

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
        self.threshold_method = self.evaluator_kwargs.threshold_method

        self.sigmas           = self.metric_kwargs.sigmas
        self.kernel_type      = self.metric_kwargs.kernel_type
        self.metric_type      = self.metric_kwargs.metric_type
        self.dyadic_orders    = self.metric_kwargs.dyadic_orders
        self.lambd            = self.metric_kwargs.lambd

        # Functions specific to class
        self.metric           = self.initialise_metric()
        self.path_evaluator   = self.lambda_path_evaluator

        super(AutoEvaluator, self).__init__(
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
        kernel_type   = self.kernel_type
        metric_type   = self.metric_type
        sigmas        = self.sigmas
        dyadic_orders = self.dyadic_orders
        lambd         = self.lambd
        length        = self.get_final_path_length()

        # Define which ambient kernel is used on the state space
        stk2str = "RBFKernel" if kernel_type.lower() == "rbf" else "LinearKernel"
        stk_kwargs_0 = {"sigma": sigmas[0]} if kernel_type.lower() == "rbf" else {}
        static_kernel_0  = getattr(sigkernel, stk2str)(**stk_kwargs_0)

        if len(self.sigmas) > 1:
            stk_kwargs_1 = {"sigma": sigmas[1], "add_time": length-1} if kernel_type.lower() == "rbf" else {"add_time": length-1}
            static_kernel_1 = getattr(sigkernel, stk2str)(**stk_kwargs_1)
            signature_kernel = sigkernel.SigKernel(
                static_kernel   = [static_kernel_0, static_kernel_1],
                dyadic_order    = dyadic_orders
            )

            def metric(x, y): return signature_kernel.compute_mmd(wrapper(x), wrapper(y), lambda_=lambd, order=2)
            print("Metric initialised. MMD2, sigma = {:.4f}, {:.4f}, dyadic_orders = {}, {}".format(
                sigmas[0], sigmas[1], dyadic_orders[0], dyadic_orders[1])
            )

        else:
            signature_kernel = sigkernel.SigKernel(static_kernel=static_kernel_0, dyadic_order=dyadic_orders[0])
            _sigma_txt = f"sigma = {sigmas[0]:.4f}" if kernel_type.lower() == "rbf" else ""
            # Define the metric (method) for scoring
            if metric_type == "mmd":
                def metric(x, y): return signature_kernel.compute_mmd(wrapper(x), wrapper(y), order=1)
                print(f"Metric initialized. MMD1, kernel = {kernel_type}, dyadic_order = {dyadic_orders[0]}")
            else:
                def metric(x, y):
                    X = wrapper(x)
                    y = wrapper(y)
                    assert not y.requires_grad, "the second input should not require grad"

                    K_XX = signature_kernel.compute_Gram(X, X, sym=True)
                    K_Xy = signature_kernel.compute_Gram(X, y, sym=False)

                    K_XX_m = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / (
                                K_XX.shape[0] * (K_XX.shape[0] - 1.))

                    return K_XX_m - 2. * torch.mean(K_Xy)
                print(f"Metric initialized. SigKer Scoring Rule, kernel = {kernel_type}, dyadic_order = {dyadic_orders[0]}")

        return metric

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
        alpha            = self.alpha_value
        prior_dist       = np.zeros(n_scores)

        if kwargs == {}:
            lags = self.lags
        else:
            lags = kwargs.get("lags")

        sub_paths             = get_sub_paths(test_path, n_steps, offset)
        mmd_paths             = get_grouped_paths(sub_paths, n_paths)
        sub_paths_transformed = self.path_transformer.transform_paths(sub_paths.copy())
        mmd_paths_transformed = get_grouped_paths(sub_paths_transformed, n_paths)

        # Init scores array
        fb         = -min(lags)
        num_lags   = len(lags)
        scores     = np.zeros(mmd_paths_transformed.shape[0]-fb)
        alphas     = np.zeros(mmd_paths_transformed.shape[0]-fb)
        indexes    = mmd_paths[fb:, -1, -1, 0]
        pbitr      = mmd_paths_transformed[fb:].copy()

        for i, paths in enumerate(tqdm(pbitr, position=0)):
            this_score = 0

            for lag in lags:
                path_ind = i + lag + fb
                this_score += metric(paths[weights], mmd_paths_transformed[path_ind][weights]).item()

            final_score = this_score/num_lags
            scores[i]   = final_score

            # Handle prior stuff
            prior_dist[i % n_scores] = final_score
            if i <= n_scores:
                alphas[i] = np.inf
            else:
                if self.threshold_method.lower() == "gamma":
                    thresh     = GammaFitter(prior_dist).ppf(alpha)
                else:
                    thresh = sorted(prior_dist)[int(alpha*n_scores)]
                alphas[i]  = thresh

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

    def generate_distance_matrix(self, test_path: np.ndarray):
        """
        Generates distance matrix inherited from Processor class

        :param test_path:   Test path to group and evaluate
        :return:            Distance matrix D
        """

        n_steps = self.n_steps
        offset  = self.offset
        n_paths = self.n_paths

        sub_paths = get_sub_paths(test_path, n_steps, offset)
        sub_paths_transformed = self.path_transformer.transform_paths(sub_paths.copy())
        mmd_paths_transformed = get_grouped_paths(sub_paths_transformed, n_paths)

        return self._generate_distance_matrix(mmd_paths_transformed)
