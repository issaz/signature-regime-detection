import os
from typing import Callable, List

import numpy as np
from higherOrderKME import sigkernel
from tqdm import tqdm
import torch

from src.testing import TestConfig
from src.testing.discriminators.Processor import Processor
from src.testing.discriminators.config import ProcessorConfig
from src.utils.auxiliary_classes.PathTransformer import PathTransformer
from src.utils.helper_functions.data_helper_functions import reweighter
from src.utils.helper_functions.global_helper_functions import get_project_root, mkdir
from src.utils.helper_functions.test_helper_functions import get_sub_paths, get_grouped_paths


class GeneralMMDDetector(Processor):
    """
    Instance of Processor class that uses the Sig-MMD approach to detect changes in regime.
    """

    def __init__(self, beliefs: np.ndarray, path_details: str, path_transformer: PathTransformer,
                 processor_config: ProcessorConfig, test_config: TestConfig):
        """
        Initialises MarkovMMDDetector object as an instance of Processor object

        :param beliefs:             k x n x l array. Bank of paths to build prior over.
        :param path_details:        String. Details of the bank of paths (name, parameters)
        :param path_transformer:    PathTransformer. Object which transforms paths given a list of transformations
        :param processor_config:    ProcessorConfig. Contains algorithm-specific configuration requirements
        :param test_config:         TestConfig. Contains path-specific configuration requirements
        """

        # Extract from input
        self.beliefs          = beliefs
        self.path_details     = path_details
        self.path_transformer = path_transformer

        self.algorithm_name   = "generalmmddetector"
        self.processor_config = processor_config
        self.algorithm_kwargs = getattr(self.processor_config, "{}_kwargs".format(self.algorithm_name))
        self.metric_kwargs    = self.algorithm_kwargs.metric_kwargs
        self.evaluator_kwargs = self.algorithm_kwargs.evaluator_kwargs

        # Config variables
        self.n_steps          = test_config.n_steps
        self.n_paths          = test_config.n_paths
        self.offset           = test_config.offset
        self.weight_factor    = test_config.weight_factor
        self.weights          = reweighter(self.n_paths, self.weight_factor)
        self.n_tests          = self.algorithm_kwargs.n_tests
        self.n_evaluations    = self.algorithm_kwargs.n_evaluations

        self.kernel_type      = self.metric_kwargs.kernel_type
        self.metric_type      = self.metric_kwargs.metric_type
        self.sigmas           = self.metric_kwargs.sigmas
        self.dyadic_orders    = self.metric_kwargs.dyadic_orders
        self.lambd            = self.metric_kwargs.lambd

        # Transform paths
        self.transformed_paths = self.transform_path_bank()

        # Functions specific to class
        self.metric           = self.initialise_metric()
        self.prior_generator  = self.lambda_prior_generator
        self.path_evaluator   = self.lambda_path_evaluator

        super(GeneralMMDDetector, self).__init__(
            beliefs           = self.beliefs,
            path_details      = self.path_details,
            algorithm_type    = self.algorithm_name,
            metric            = self.metric,
            prior_generator   = self.prior_generator,
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
                print(f"Metric initialized. MMD1, sigma={sigmas[0]}, kernel = {kernel_type}, dyadic_order = {dyadic_orders[0]}")
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

    def lambda_prior_generator(self) -> (np.ndarray, float):
        """
        Function which generates prior distribution, or loads it, if previously calculated

        :return:            (Array, float). MMD scores and critical value c_alpha.
        """

        metric             = self.metric
        pt_args            = self.path_transformer.transformations.items()
        n_tests            = self.n_tests
        n_steps            = self.n_steps
        paths              = self.transformed_paths
        pathwise_sig       = self.path_transformer.compute_pathwise_signatures
        pathwise_sig_order = self.path_transformer.signature_order
        beliefs            = self.beliefs

        # Number of paths to generate from, changes if there are weights
        n_paths = len(self.weights)

        if len(self.sigmas) > 1:
            metric_type = "mmd2"
        elif self.metric_type == "scoring":
            metric_type = "scoring"
        else:
            metric_type = "mmd1"

        path_ = get_project_root().as_posix() + "/data/mmd_scores/{}_{}_d_{}_p_{}_l_{}_t_{}_s_{}_{}_ps_{}_le_{}.npy".format(
            metric_type,
            "_".join([str(detail) for detail in self.path_details]),
            beliefs.shape[-1]-1,
            n_paths,
            n_steps,
            n_tests,
            "_".join([str(np.round(s, 4)).replace(".", "") for s in self.sigmas]),
            "".join([str(int(v[0])) for _, v in pt_args]),
            pathwise_sig,
            pathwise_sig_order
        )

        path_exists = os.path.exists(path_)

        if path_exists and not self.overwrite_prior:
            metric_scores = np.load(path_, allow_pickle=True)
        else:
            metric_scores = np.zeros(shape=(beliefs.shape[0], n_tests))

            if self.metric_type == "mmd":
                for k, bank in enumerate(self.transformed_paths):
                    path_bank_size = bank.shape[0]
                    ii = np.random.randint(0, path_bank_size, size=(n_tests, 2, n_paths))

                    for i, randi in tqdm(enumerate(ii), position=0):
                        metric_scores[k, i] = metric(bank[randi[0]], bank[randi[1]])
            else:
                for k, bank in enumerate(self.transformed_paths):
                    path_bank_size = bank.shape[0]
                    ii = np.random.randint(0, path_bank_size, size = (n_tests, n_paths))
                    jj = np.random.randint(0, path_bank_size, size=(n_tests, 1))

                    for i, (randi, randj) in tqdm(enumerate(zip(ii, jj)), position=0):

                        metric_scores[k, i] = metric(bank[randi], bank[randj])

            np.save(path_, metric_scores, allow_pickle=True)

        # Get critical value
        c_alpha = np.zeros((beliefs.shape[0], 1))
        for i, score_vector in enumerate(metric_scores):

            c_alpha[i] = np.sort(score_vector)[int(self.alpha_value*n_tests)]

        return metric_scores, c_alpha

    def lambda_path_evaluator(self, test_path: np.ndarray, **kwargs) -> np.ndarray:
        """
        Path evaluator function for GeneralMMDDetector object

        :param test_path:   Path to be evaluated

        :return:            Scores under given metric and their corresponding indexes relative to the path provided
        """
        if kwargs is None:
            evaluation = "total"
        else:
            evaluation = kwargs.get("evaluation").lower()

        # Init variables
        n_steps          = self.n_steps
        offset           = self.offset
        n_paths          = self.n_paths
        weights          = self.weights
        n_evaluations    = self.n_evaluations
        metric           = self.metric
        t_path_bank      = self.transformed_paths
        n_weighted_paths = self.weights.shape[0]

        metric_type      = self.metric_type

        indexes          = np.array([])

        # Extract sub-paths from path to be evaluated and transform them
        sub_paths             = get_sub_paths(test_path, n_steps, offset)
        sub_paths_transformed = self.path_transformer.transform_paths(sub_paths.copy())

        if evaluation == "total":

            if metric_type.lower() == "mmd":
                # Extract MMD paths
                mmd_paths = get_grouped_paths(sub_paths_transformed, n_paths)

                # Init scores array
                scores = np.zeros(shape=(len(t_path_bank), mmd_paths.shape[0]))

                # Get random integers for sampling from path bank, number of integers given by number of evaluations
                for k, bank in enumerate(t_path_bank):
                    ii = np.random.randint(0, bank.shape[0], size=(mmd_paths.shape[0], n_evaluations, n_weighted_paths))

                    for i, paths in enumerate(tqdm(mmd_paths, position=0)):
                        # Eval multiple times for each atom
                        for ind in ii[i]:
                            these_paths = bank[ind]
                            scores[k, i] += metric(these_paths, paths[weights])

                        # Normalise by number of evaluations
                        scores[k, i] /= n_evaluations

                    # Extract indices of MMD scores for plotting
                    indexes = np.array([sub_paths[n_paths-1:, -1, 0]])

            else:
                # Init scores array
                scores = np.zeros((len(t_path_bank), sub_paths.shape[0]))

                for k, bank in enumerate(t_path_bank):
                    ii = np.random.randint(0, bank.shape[0], size=(sub_paths.shape[0], n_evaluations, n_weighted_paths))

                    for i, path in tqdm(enumerate(sub_paths_transformed), position=0):
                        for ind in ii[i]:
                            these_paths = bank[ind]
                            scores[k, i] += metric(these_paths, [path])

                        scores[k, i] /= n_evaluations

                    indexes = np.array([sub_paths[:, -1, 0]])

        elif evaluation == "increment":

            # Ignore first x% of paths
            num_sub_paths = sub_paths_transformed.shape[0]
            path_threshold = int(self.evaluator_kwargs.pct_ignore*num_sub_paths + 1)

            # Init scores array
            scores = np.zeros(shape=(t_path_bank.shape[0], num_sub_paths))

            for i in tqdm(range(path_threshold, num_sub_paths), position=0):

                this_mmd = 0

                if i < n_paths:
                    # Small vs large sample
                    these_paths = sub_paths_transformed[:i]
                else:
                    # Can take entire sample size
                    these_paths = sub_paths_transformed[i-n_paths:i]

                for k, bank in enumerate(t_path_bank):
                    for _ in range(n_evaluations):
                        # Calculate MMD for each n evaluations
                        ii = np.random.randint(0, bank.shape[0], n_paths)
                        prior_paths = bank[ii]
                        this_mmd += metric(prior_paths, these_paths)

                    scores[k, i] = this_mmd/n_evaluations

            indexes = np.array([sub_paths[:, -1, 0]])
            scores  = np.array(scores)
        else:
            indexes = []
            scores = []

        return np.concatenate([indexes, scores])

    # Other evaluation functions
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
