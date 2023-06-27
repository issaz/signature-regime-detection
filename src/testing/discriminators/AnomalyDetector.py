import os
from typing import List, Callable

import iisignature
import numpy as np
import torch
from tqdm import tqdm

from src.testing import TestConfig
from src.testing.discriminators.Processor import Processor
from src.testing.discriminators.config import ProcessorConfig
from src.utils.auxiliary_classes.PathTransformer import PathTransformer
from src.utils.helper_functions.global_helper_functions import get_project_root, mkdir
from src.utils.helper_functions.signature_helper_functions import shuffle_product, tuples_to_strings, all_words
from src.utils.helper_functions.test_helper_functions import get_sub_paths


class AnomalyDetector(Processor):
    """
    Implementation of the process of anomaly detection outlined in "Anomaly Detection on Streamed Data"
    """

    def __init__(self, beliefs: np.ndarray, path_details: str, path_transformer: PathTransformer,
                 processor_config: ProcessorConfig, test_config: TestConfig):
        """
        Initialises the AnomalyDetector class as an instance of the Processor object

        :param beliefs:             Array. Bank of paths to build prior over
        :param path_details:        String. List of path details for indexing and saving purposes
        :param path_transformer:    Instance of PathTransformer object with desired path transformations
        :param processor_config:    Instance of ProcessorConfig with algorithm-specific configurations
        :param test_config:         Instance of TestConfig for path evaluation purposes
        """

        self.beliefs              = beliefs
        self.path_details         = path_details

        self.path_transformer     = path_transformer

        self.algorithm_name       = "anomalydetector"
        self.processor_config     = processor_config
        self.test_config          = test_config
        self.algorithm_kwargs     = getattr(processor_config, "{}_kwargs".format(self.algorithm_name))
        self.signature_type       = self.algorithm_kwargs.signature_type
        self.signature_depth      = self.algorithm_kwargs.signature_depth
        self.pct_path_bank        = self.algorithm_kwargs.pct_path_bank
        self.device               = self.processor_config.device

        # Test values
        self.n_steps              = self.test_config.n_steps
        self.offset               = self.test_config.offset

        # Transform paths
        self.transformed_paths    = self.transform_path_bank()[0]  # Only ever assume one set of beliefs

        # Class specific objects
        self.D1                   = None
        self.D2                   = None
        self.A                    = None
        self.invA                 = None
        self.corpus_signatures    = None
        self.corpus_signatures_2N = None

        self.metric               = self.initialise_metric()
        self.prior_generator      = self.lambda_prior_generator
        self.path_evaluator       = self.lambda_path_evaluator

        super(AnomalyDetector, self).__init__(
            beliefs          = self.beliefs,
            path_details     = self.path_details,
            algorithm_type   = self.algorithm_name,
            metric           = self.metric,
            prior_generator  = self.prior_generator,
            path_evaluator   = self.path_evaluator,
            path_transformer = self.path_transformer,
            config           = self.processor_config
        )

    # Processor-specific functions
    def initialise_metric(self) -> Callable:
        """
        Initialises conformance metric

        :return:    Callable function which acts as metric
        """

        # Generate initial objects
        self.D1, self.D2          = self.split_distribution()
        self.corpus_signatures_2N = self.get_corpus_signatures(self.D1, 2)
        self.corpus_signatures    = self.get_corpus_signatures(self.D1, 1)

        # Calibrate variance norm
        print("Calibrating variance norm...")
        self.calibrate_variance_norm()

        return self.conformance

    def lambda_prior_generator(self) -> (np.ndarray, float):
        """
        Generates prior distribution and critical value

        :return:
        """
        pt_args = self.path_transformer.transformations.items()

        data_path = get_project_root().as_posix() + "/data/anomaly_scores/{}_d_{}_s_{}_l_{}_t_{}_o_{}_logsig_{}.npy".format(
            self.path_details[0],
            self.beliefs.shape[-1] - 1,
            int(self.pct_path_bank*self.beliefs.shape[0]),
            self.n_steps,
            "".join([str(int(v[0])) for _, v in pt_args]),
            self.signature_depth,
            self.signature_type == "log_signature"
        )

        path_exists = os.path.exists(data_path)

        if path_exists and not self.overwrite_prior:
            distribution = np.load(data_path, allow_pickle=True)
        else:
            distribution = np.array([self.metric(path) for path in tqdm(self.D2, position=0, leave=True)])
            np.save(data_path, distribution, allow_pickle=True)

        sorted_scores = np.sort(distribution)
        c_alpha       = sorted_scores[int(len(sorted_scores)*self.alpha_value)]

        return np.array([sorted_scores]), c_alpha

    def lambda_path_evaluator(self, test_path: np.ndarray) -> List[np.ndarray]:
        """
        Evaluates segments of a path using the framework provided in "Anomaly detection on streamed data"

        :param test_path:   Path to evaluate

        :return:            Scores under given metric and their corresponding indexes relative to the path provided
        """

        # Init variables
        n_steps               = self.n_steps
        offset                = self.offset

        # Transform appropriate paths
        sub_paths             = get_sub_paths(test_path, n_steps, offset)
        sub_paths_transformed = self.path_transformer.transform_paths(sub_paths.copy())
        conformance_scores    = np.zeros(sub_paths_transformed.shape[0])

        # Calculate conformance scores
        for i, path in enumerate(tqdm(sub_paths_transformed, position=0, leave=True)):
            conformance_scores[i] = self.metric(path)

        indexes = sub_paths[:, -1, 0]

        return np.c_[indexes, conformance_scores].T

    # Class-specific functions
    def split_distribution(self) -> List[np.ndarray]:
        """
        Splits a corpus of paths into two disjoint subsets.

        :return:    List of corpus of paths that have been split. Already transformed by Processor object
        """
        path_bank  = self.transformed_paths
        num_values = int(path_bank.shape[0]*self.pct_path_bank)

        random_indexes = np.random.choice(path_bank.shape[0], num_values, replace=False)
        idx, idy = np.split(random_indexes, 2)

        return [path_bank[idx], path_bank[idy]]

    def get_corpus_signatures(self, paths: np.ndarray, scale: int) -> torch.tensor:
        """
        Returns signatures associated to corpus of paths

        :param paths:   Corpus of paths
        :param scale:   Additional argument to calculate higher order signatures than the default
        :return:        n x l x 1+d+d^2+... tensor of corpus signatures, where d is the dimension of the paths
        """

        corpus_signatures = self.calculate_signature(paths, int(scale*self.signature_depth))

        return corpus_signatures

    def calibrate_variance_norm(self) -> None:
        """
        Calculates required values to output the signature variance norm as given in
        Propositions 3.1, 3.2: ``Anomaly Detection on Streamed Data''.
        """

        dim                = self.beliefs.shape[-1]
        order              = self.signature_depth
        num_terms          = int((np.power(dim, order + 1)-1)/(dim - 1)) - 1
        expected_signature = torch.mean(self.corpus_signatures_2N, 0, dtype=torch.float64)

        input_words = all_words(dim, 2*order)
        words       = tuples_to_strings(input_words)[1:]

        # Generate matrix A
        shuffles = self.wrapper(
            [[shuffle_product(i, j, words) for i in range(num_terms)] for j in range(num_terms)]
        )

        self.A    = torch.einsum("ijp,p->ij", shuffles, expected_signature)
        self.invA = torch.inverse(self.A)

    def variance_norm(self, w: torch.Tensor) -> float:
        """
        Calculates the variance norm associated to a corpus of data as calibrated.

        :param w: Vector in R^{d_N} to calculate variance norm
        :return:  Variance norm value
        """
        assert self.invA is not None, "ERROR: Variance norm not calibrated"

        return np.abs(torch.matmul(w, torch.matmul(self.invA, w.T)).item())

    def conformance(self, path: np.ndarray) -> float:
        """
        Gives the conformance score of an element w to a corpus of datum. See Definition 1.2, Anomaly Detection on
        Streamed Data.

        :param path:  Path to be compared to corpus
        :return:      Conformance score
        """

        variance_norm  = self.variance_norm
        path_signature = self.calculate_signature(np.expand_dims(path, 0), self.signature_depth)
        conformance = 0

        for sig in self.corpus_signatures:
            this_score = variance_norm(path_signature - sig)
            conformance = max([conformance, this_score])

        return conformance

    def calculate_signature(self, x, order):
        sig = iisignature.sig(x, order)
        return self.wrapper(sig)
