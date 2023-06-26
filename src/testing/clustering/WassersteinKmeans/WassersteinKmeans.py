import copy
import random

import numpy as np

from src.testing.clustering.config import ClusterConfig


class WassersteinKmeans(object):
    """
    Simple WassersteinKmeans clustering object

    """
    def __init__(self, data: np.ndarray, config: ClusterConfig):
        self.n_clusters                         = config.n_clusters

        self.algorithm_kwargs                   = getattr(config, "wassersteinkmeans_kwargs")
        self.window_length                      = self.algorithm_kwargs.window_length
        self.overlap                            = self.algorithm_kwargs.overlap
        self.wp                                 = self.algorithm_kwargs.wp

        self.cluster_data                       = data.copy()
        self.cluster_data[1:, 1]                = np.diff(np.log(data[:, 1]))
        self.cluster_data[0, 1]                 = 0.

        self.partitions, self.partition_indexes = self.partition_data()
        self.centroids                          = self.initialize_centroids()
        self.metric                             = self.initialize_metric()
        self.aggregator                         = self.initialize_aggregator()
        self.loss                               = self.initialize_loss()
        self.rank_map                           = None

    def partition_data(self):
        cluster_data  = self.cluster_data
        sample_length = self.window_length
        overlap       = self.overlap

        N = len(cluster_data) - 1
        offset = sample_length - overlap

        M = int((N - sample_length - 1) / offset) + 1

        # Partition data
        partitions_init = {
            i + 1: cluster_data[1 + i * offset:1 + i * offset + sample_length, 1]
            for i in range(0, M)
        }

        partition_indexes_init = {
            i + 1: cluster_data[1 + i * offset:1 + i * offset + sample_length, 0]
            for i in range(0, M)
        }

        partitions = dict(
            filter(lambda elem: len(elem[1]) == sample_length, partitions_init.items())
        )

        partition_indexes = dict(
            filter(lambda elem: len(elem[1]) == sample_length, partition_indexes_init.items())
        )

        return partitions, partition_indexes

    def initialize_centroids(self):
        partitions = self.partitions
        n_clusters = self.n_clusters

        rand_indexes = random.sample(list(partitions.keys()), n_clusters)

        return {i+1: partitions[k] for i, k in enumerate(rand_indexes)}

    def assign_data(self) -> np.ndarray:

        centroids          = self.centroids
        partitions         = self.partitions
        metric             = self.metric

        results = np.zeros((len(partitions), self.n_clusters + 1))

        for k, v in centroids.items():
            results[:, k-1] = np.array([metric(x, v) for x in partitions.values()])

        results[:, -1] = np.argmin(results[:, :-1], axis=1) + 1

        return results

    def update(self, results: np.ndarray) -> None:

        cluster_partitions = np.array(list(self.partitions.values()))

        for i in self.centroids.keys():
            filtered_dists = cluster_partitions[np.where(results[:, -1]  == i)[0]]
            self.centroids[i] = self.aggregator(filtered_dists)

    def calculate_centroids(self, tol=np.nextafter(0, 1), max_iters=100, verbose=False) -> None:

        assign_data = self.assign_data
        update      = self.update
        loss        = self.loss

        n_iter = 0

        while True:
            old_cent = copy.deepcopy(self.centroids)
            update(assign_data())

            val = loss(
                old_cent.values(),
                self.centroids.values()
            )

            n_iter += 1

            if (val < tol) or (n_iter > max_iters):
                if verbose:
                    print("Convergence achieved after {} steps".format(n_iter))
                break

        self.determine_ranking_map()
        self.centroids = {self.rank_map[k]: v for k, v in self.centroids.items()}

    # Metric and aggregator
    def initialize_metric(self):
        return lambda x, y, ex=self.wp: self.wasserstein_distance(x, y, ex)

    def initialize_aggregator(self):
        def wasserstein_barycenter(dists, p):
            s_dists = np.array([np.sort(d, kind='quicksort') for d in dists])

            if p == 1:
                return np.median(s_dists, axis=0)
            else:
                return np.mean(s_dists, axis=0)

        return lambda d, ex=self.wp: wasserstein_barycenter(d, ex)

    def initialize_loss(self):
        def loss(x, y, p):
            return np.sum([self.wasserstein_distance(xi, yi, p) for xi, yi in zip(x, y)])

        return lambda x,y,ex=self.wp: loss(x, y, ex)

    @staticmethod
    def wasserstein_distance(x, y, p):
        expo = 1./p
        return np.power(np.mean(np.power(np.abs(np.sort(x) - np.sort(y)), p)), expo)

    # Helper functions
    @staticmethod
    def sort_dict_by_f(d: dict, f, **kwargs) -> list:
        """
        Sorts a dictionary by a function :param f:
        :param d:       Dictionary to be sorted
        :param f:       Function to sort over. Types in the dict and main argument of f need to match.
        :param kwargs:  Optional arguments to pass to function.
        :return:        Sorted list of values from dictionary :param d:
        """

        try:
            return sorted({k: f(v, **kwargs) for k, v in d.items()}.items(), key=lambda x: x[1], reverse=False)
        except TypeError:
            return sorted({k: f(v) for k, v in d.items()}.items(), key=lambda x: x[1], reverse=False)

    def sort_centroids_by_var(self) -> list:
        """
        Sorts centroids by variance (for initialisation)

        :return: Sorted centroids as a list
        """

        centroids = copy.deepcopy(self.centroids)

        return self.sort_dict_by_f(centroids, np.var)

    def determine_ranking_map(self):
        """
        Determines labelling of obtained centroids

        :return:    None
        """
        var_sorted = self.sort_centroids_by_var()

        ranks = np.arange(0, self.n_clusters) + 1

        self.rank_map = {k[0]: v for k, v in zip(var_sorted, ranks)}

    def assign_labels(self) -> np.ndarray:
        """
        Assigns labels to each partition

        :return: Array assigning labels to each element of partitions dictionary
        """

        labels  = self.assign_data()[:, -1]
        indexes = np.array(list(self.partition_indexes.values()))[:, -1]

        return np.vstack([indexes, labels])
