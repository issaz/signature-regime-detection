import numpy as np
import torch


def calculate_separation_score(P: np.ndarray, Q: np.ndarray, quantiles: int):
    """
    Returns the separation score between two empirical distributions. Requires both distributions have the same number
    of atoms.

    :param P:           First distribution to calculate score of.
    :param Q:           Second distribution to calculate score of.
    :param quantiles    Number of quantiles to calculate separation score over. True is when number of quantiles is the
                        same as the number of atoms.
    :return:            Separation score. A score of 1 implies the distributions are totally separate. A score of 0
                        implies that they are the same.
    """

    assert np.shape(P) == np.shape(Q)

    p_quantiles = np.quantile(P, np.linspace(0, 1, quantiles))
    q_quantiles = np.quantile(Q, np.linspace(0, 1, quantiles))

    p_conf = np.array([np.sum(p_quantiles < q) for q in q_quantiles])/quantiles

    return np.log(1/(1-np.mean(p_conf) + 0.000001))


def separation_score_torch(quantiles: int):
    """
    Separation scores as implemented in pytorch.
    :param quantiles:   Number of quantiles to calculate separation score over
    :return:            Loss function with backward method
    """
    def loss(y_pred, y_true):
        pass

    return loss


def wasserstein_loss_torch(p: int):
    """
    Wasserstein distance between histograms for use in keras

    :param p    Wasserstein exponent
    :return:    Loss function evaluation using pytorch
    """

    def loss(y_pred, y_true):
        """
        Must accept two torch tensors with requires_grad == True
        :param y_pred:  First tensor
        :param y_true:  Second tensor
        :return:        Wasserstein distance between empirical measures
        """

        yp_sorted = torch.sort(y_pred).values
        yt_sorted = torch.sort(y_true).values

        return torch.pow(torch.mean(torch.pow(torch.abs(yp_sorted - yt_sorted), p)), 1/p)

    return loss


def calculate_separation_score_v2(P, Q, n_bins=10):
    min_range, max_range = np.min([P, Q]), np.max([P, Q])

    P_est = np.histogram(P, bins=n_bins, range=(min_range, max_range), density=True)
    Q_est = np.histogram(Q, bins=n_bins, range=(min_range, max_range), density=True)

    mesh = np.diff(P_est[1])[0]
    P_score = 0
    Q_score = 0

    for px, py in zip(P_est[0], Q_est[0]):
        if (px > 0) and (py > 0):
            P_score += px * mesh
            Q_score += py * mesh

    return 1 - np.mean([P_score, Q_score])


