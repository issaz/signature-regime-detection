import numpy as np

from sklearn.cluster import AgglomerativeClustering

from src.testing.clustering.config import ClusterConfig


class HierarchicalClusterer(object):
    """
    Object for clustering general distance matrices
    """
    def __init__(self, distance_matrix: np.ndarray, config: ClusterConfig):
        """
        Constructor object

        :param distance_matrix:     Matrix of distances between elements: must be square
        """
        assert len(distance_matrix.shape) == 2, "ERROR: Not a distance matrix"
        assert distance_matrix.shape[0] == distance_matrix.shape[-1], "ERROR: Non-square distance matrix"

        self.distance_matrix  = distance_matrix
        self.num_elements     = distance_matrix.shape[0]
        self.n_clusters       = config.n_clusters
        self.algorithm_kwargs = getattr(config, "hierarchicalclusterer_kwargs")
        self.linkage          = self.algorithm_kwargs.linkage

    def get_hierarchical_clusters(self, connectivity_matrix=None) -> np.ndarray:
        """
        Obtains hierarchical clusters from sklearn package

        :param connectivity_matrix:     Optional. Matrix of connections
        :return:
        """

        n_clusters = self.n_clusters
        linkage    = self.linkage

        # Define named arguments
        names  = ("n_clusters", "affinity", "linkage", "connectivity")
        values = (n_clusters, "precomputed", linkage, connectivity_matrix)
        kwargs = {k: v for k, v in zip(names, values) if v}

        # Obtain clusters
        clusters = AgglomerativeClustering(**kwargs).fit(self.distance_matrix)
        labels   = clusters.labels_

        return labels
