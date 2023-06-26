from functools import reduce, partial

import torch
import iisignature
import numpy as np
from tqdm import tqdm

import path_transformations
from src.utils.auxiliary_classes.PathTransformer import PathTransformerConfig


class PathTransformer(object):
    """
    Applies succession of path transformations as outlined by a dictionary.

    Pathwise signature transform needs to be done separately due to computational reasons, i.e. undocking and
    re-docking from GPU to CPU to GPU for single path instances is expensive and drastically slows down the pathwise
    signature process.
    """

    def __init__(self, config: PathTransformerConfig):
        self.transformations             = config.transformations
        self.compute_pathwise_signatures = config.compute_pathwise_signature_transform
        self.signature_order             = config.signature_order

        # Set transformations ex pathwise signature transform
        self.phi = self.set_phi()

    def set_phi(self):

        these_functions = [lambda x: x, lambda x: x]

        ordered_transformations = {k: v for k, v in sorted(
            self.transformations.items(), key=lambda item: item[1][1]
        ) if v[0]}

        def compose(*functions):
            def compose2(f, g):
                return lambda x: g(f(x))

            return reduce(compose2, functions, lambda x: x)

        # Add functions to list
        for name, (appl, _, kwargs) in ordered_transformations.items():
            if appl:
                this_func = getattr(path_transformations, name)
                these_functions.append(partial(this_func, **kwargs))

        return compose(*these_functions)

    def transform_paths(self, paths: np.ndarray, verbose=True) -> np.ndarray:
        """
        Applied to set of paths instead of one individual path

        :param paths:   Bank of paths to transform
        :param verbose: Print iterator
        :return:        Transformed paths
        """

        phi = self.phi
        path_bank = []

        itr = tqdm(paths.copy(), position=0, leave=True) if verbose else paths.copy()

        for p in itr:
            path_bank.append(phi(p))

        path_bank = np.array(path_bank)

        if self.compute_pathwise_signatures:
            path_bank = self.pathwise_signature_transform(path_bank)

        return path_bank

    def signature_transform(self, paths: np.ndarray) -> np.ndarray:
        """
        Returns the signature transformation of the given path up to the given order.

        :param paths:    Paths to take signature of
        :return:         Truncated signature of order N
        """
        assert len(paths.shape) == 3, "ERROR: paths not of dimension n x l x d"

        # Convert path to torch tensor for use with signatory package
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        # Convert to 1 x l x d torch tensor
        tensor_path = torch.tensor(paths, dtype = torch.float32, device=device)

        # Compute signature
        signatures = iisignature.sig(tensor_path, self.signature_order)

        # Sadly we have to shift back to numpy
        if torch.cuda.is_available():
            return signatures.cpu().numpy()
        else:
            return signatures.numpy()

    def pathwise_signature_transform(self, paths: np.ndarray) -> np.ndarray:
        """
        Returns the pathwise signature transform of a given path. That is, returns the path evolving in the (truncated)
        tensor algebra space associated to the truncated signature of order N.

        :param paths:    Path to compute transform over
        :return:        Pathwise signature transform
        """
        assert len(paths.shape) == 3, "ERROR: paths need to be of dimension n x l x d"

        n, length, dim = paths.shape
        sig_length = sum(pow(dim, i+1) for i in range(self.signature_order))
        res = np.zeros((n, length-1, sig_length))

        for i in range(1, length):
            res[:, i-1, :] = self.signature_transform(paths[:, :i+1, :])

        return res
