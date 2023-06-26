import numpy as np


def get_sub_paths(path: np.ndarray, n_steps: int, offset: int) -> np.ndarray:
    """
    Gets all sub-paths of length n_steps offset by the offset parameter.

    :param path:        Path to get sub-paths from
    :param n_steps:     Length of each sub-path
    :param offset:      Overlap offset. If 0, paths are all distinct

    :return:            N x n_steps x d tensor of sub-paths, where N is the total number of sub-paths extracted
    """

    total_sub_paths = int(len(path) / (n_steps - offset))

    sub_paths = np.array([
        path[i * (n_steps - offset):i * (n_steps - offset) + n_steps] for i in range(total_sub_paths)
    ])

    return sub_paths


def get_grouped_paths(sub_paths: np.ndarray, n_paths: int) -> np.ndarray:
    """
    Collects sub-paths into equal groups of size n_paths, for calculation using the MMD.

    :param sub_paths:   N x n_steps x d tensor of sub-paths
    :param n_paths:     Number of paths to include in each group
    :return:            M x n_paths x n_steps x d tensor, where M is the total number of groups of size n_paths
                        that can be extracted from the sub_paths object
    """
    total_mmd_paths = int(sub_paths.shape[0] - n_paths + 1)

    return np.array([sub_paths[i:i+n_paths] for i in range(total_mmd_paths)])


def get_memberships(grouped_paths: np.ndarray) -> list:
    """
    Gets the sub-path membership indices given a bank of grouped paths.

    :param grouped_paths:   Bank of paths to compute memberships of
    :return:                Memberships as a ragged list of lists
    """
    shape_vec = grouped_paths.shape
    total_paths = shape_vec[0] + shape_vec[1] - 1

    return [[i for i in range(max(k + 1 - shape_vec[1], 0), min(k + 1, shape_vec[0]))] for k in range(total_paths)]


def get_alphas(membership_vector: list, results: np.ndarray, c_alpha: np.ndarray) -> np.ndarray:
    """
    Gets the plot alpha values of a sub-path given a vector of test results and a memberships list.

    :param membership_vector:   Ragged list of lists corresponding to group memberships
    :param results:             Array of test scores from a given Processor instance
    :param c_alpha:             Critical value associated to Processor instance
    :return:                    Array of alphas for each sub-path corresponding to memberships vector
    """

    mmd_scores = results[1:, :]
    alphas = np.zeros(shape=(results.shape[0]-1, len(membership_vector)))

    for i, m in enumerate(membership_vector):
        if not m:
            continue

        for j, res in enumerate(mmd_scores):
            vec = res[m]
            alphas[j, i] = vec[vec > c_alpha[j]].shape[0] / len(m)

    return alphas
