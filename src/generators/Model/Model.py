import operator
from typing import List

import numpy as np
from scipy.special import gamma, gammainc

from generators.config import ModelConfig


# TODO: A better way of doing this to avoid class bloat is to define generic class and have other classes extend it.
class Model(object):
    """
    Generic model class for path generation
    """

    def __init__(self, model_type: str, params: List[List[float]], config=ModelConfig()):
        assert (model_type in config.models), "ERROR: Model not available"
        param_set = set((len(p) for p in params))
        assert len(param_set) == 1, "ERROR: Non-equal number of parameters for all models"

        # Assertion for each set of parameters
        for i, param in enumerate(params):
            op = operator.eq if model_type not in ["ar_p", "i_ar_p"] else operator.ge
            assert op(len(param), getattr(config.model_param_length, model_type)), \
                "ERROR: Incorrect number of parameters, position {}".format(i)

        # Parameters
        self.model_type         = model_type
        self.params             = np.array(params)

        # Adjust parameters in 1D case
        if len(self.params.shape) == 1:
            self.params = np.array([params])

        self.attach_volatility  = config.attach_volatility
        self.year_mesh          = config.year_mesh
        self.dim                = self.params.shape[0]
        self.path_dim           = 2*self.params.shape[0] if self.attach_volatility else self.dim
        self.correlation_matrix = np.eye(self.dim)

        # Instantiate this path sim
        tag = self.model_type

        self._path_sim = getattr(self, '_generate_{}_process'.format(tag))

    def __repr__(self):
        for dim in range(self.dim):
            return "Dimension {}: Model: {}, params: {}".format(
                dim + 1, self.model_type, ", ".join([str(p) for p in self.params[dim]])
            )

    def set_correlation_matrix(self, correlation_matrix: np.ndarray) -> None:
        """
        Sets a correlation matrix between Brownian motions, if desired.

        :param correlation_matrix:      d x d correlation matrix.
        :return:                        None
        """

        assert correlation_matrix.shape == (self.dim, self.dim), \
            "ERROR: Correlation matrix not square, wrong shape."
        assert np.diag(correlation_matrix) == np.ones(self.dim), \
            "ERROR: Diagonal elements are not equal to 1."
        assert np.triu(correlation_matrix) == np.tril(correlation_matrix).T, \
            "ERROR: Correlation matrix not symmetric"

        self.correlation_matrix = correlation_matrix

    def sim_path(self, T: float, S0: List[float], time_add=0) -> np.ndarray:
        """
        General method to simulate a path from the given Model.

        :param T:           Time to simulate till
        :param S0:          Initial stock value
        :param time_add:    Optional. Addition to time index of path
        :return:            Path sim over [0, T] with grid points given by the year mesh in ModelConfig
        """
        return self._path_sim(T, S0, time_add)

    def _generate_ar_p_process(self, T: float, S0: List[float], time_add=0) -> np.ndarray:
        """
        Generates an AR(p) process, defined by

                                X_n = \sum_{i=1}^p \phi_i X_{n-i} + \varepsilon_n,

        where (\phi_i)_{i=1}^p are the roots associated to the process, and \varepsilon_n is noise.

        :param T:           Time to simulate till
        :param S0:          Initial stock value33
        :param time_add:    Option. Addition to time index of path
        :return:            Path sim over [0, T] with grid points given by the year mesh in ModelConfig
        """

        roots, sd_n  = self.params.T[:-1], self.params.T[-1]
        roots        = np.array(roots).T
        order        = roots.shape[-1]
        grid_points  = int(self.year_mesh*T)

        path = np.zeros((grid_points, self.dim + 1))
        path[:, 0] = np.linspace(0, T, grid_points) + time_add

        for di in range(self.dim):
            noise = np.random.normal(0, np.power(sd_n[di], 2), size=(grid_points, 1))
            init_values = S0[di] + noise[:order]
            path[:order, di+1] = init_values.T[0]

            for i in range(order, grid_points):
                path[i, di+1] = roots[di] @ path[i-order:i, di+1] + noise[i]

        return path

    def _generate_i_ar_p_process(self, T: float, S0: List[float], time_add=0) -> np.ndarray:
        """
        Treats an AR(p) process as a set of returns, to obtain a stock price process

        :param T:           Time to simulate till
        :param S0:          Initial stock value
        :param time_add:    Optional. Addition to time index of path
        :return:            Integrated AR(p) process
        """

        path = self._generate_ar_p_process(T=T, S0=[0. for _ in range(self.dim)], time_add=time_add)

        for di in range(self.dim):
            path[:, di + 1] = S0[di]*(1 + path[:, di + 1]).cumprod()

        return path

    def _generate_gbm_process(self, T: float, S0: List[float], time_add=0) -> np.ndarray:
        """
        Generates a gBm with params as initialized by the instance of Model. Dynamics are given by

                                    dS_t = mu S_t dt + sigma S_t dW_t,

        where mu, sigma are the provided parameters.

        :param T:           Time to simulate till
        :param S0:          Initial stock value
        :param time_add:    Optional. Addition to time index of path
        :return:            Path sim over [0, T] with grid points given by the year mesh in ModelConfig
        """
        # Define parameters
        path_dim    = self.path_dim
        dim         = self.dim
        mu, sigma   = self.params.T
        grid_points = int(self.year_mesh*T) + 1
        res         = np.zeros((grid_points, path_dim + 1))
        timeline    = np.linspace(0, T, grid_points)

        # Generate BM drivers
        bm = self._generate_bm_path(T, self.correlation_matrix)

        # Build paths
        path = np.zeros((grid_points, dim))

        # Init vol process if necessary
        if self.attach_volatility:
            v = np.zeros((grid_points, dim))

        for i in range(dim):
            path[:, i] = S0[i]*np.exp((mu[i] - sigma[i]**2 / 2.) * timeline + sigma[i]*bm[:, i])

            if self.attach_volatility:
                v[:, i]    = np.exp(sigma[i]*bm[:, i])

        # Add time if required
        timeline += time_add

        # Return path
        res[:, 0]       = timeline
        res[:, 1:dim+1] = path

        if self.attach_volatility:
            res[:, dim+1:] = v

        return res

    def _generate_merton_process(self, T: float, S0: List[float], time_add=0) -> np.ndarray:
        """
        Simulates a Merton jump diffusion process. Dynamics are given by

                                    dS_t = mu S_t dt + sigma S_t dW_t + k_t dN_t,

        where ln(1+k_t) is distributed normally with mean mu_J and st. dev sigma_J. N_t is a Poisson point process with
        intensity lambda. Provided parameters are [mu, sigma, lambda, mu_J, sigma_J].

        :param T:           Time to simulate till
        :param S0:          Initial stock value
        :param time_add:    Optional. Addition to time index of path
        :return:            Path sim over [0, T] with grid points given by the year mesh in ModelConfig
        """

        # Define parameters
        path_dim                  = self.path_dim
        dim                       = self.dim
        mu, sigma, lam, muJ, sigJ = self.params.T
        grid_points               = int(self.year_mesh*T) + 1

        # Initialise path objects
        res       = np.zeros((grid_points, path_dim + 1))
        path      = np.zeros((grid_points, dim))
        log_jumps = np.zeros((grid_points, dim))
        timeline  = np.linspace(0, T, grid_points)
        v         = np.zeros((grid_points, dim))

        # Path components
        dt = 1/self.year_mesh
        bm = self._generate_bm_path(T, self.correlation_matrix)

        # Jumps
        for d in range(dim):
            jumps_size = np.random.normal(muJ[d], sigJ[d], size=grid_points)
            jumps_loc = np.random.poisson(lam[d]*dt, size=grid_points)
            jumps = jumps_loc*jumps_size
            log_jumps[:, d] = jumps.cumsum()

        # Build path
        for i in range(dim):
            path[:, i] = S0[i]*np.exp((mu[i] - sigma[i]**2 / 2.) * timeline + sigma[i]*bm[:, i] + log_jumps[:, i])

            if self.attach_volatility:
                v[:, i] = np.exp(sigma[i]*bm[:, i])

        # Return path
        timeline += time_add
        res[:, 0] = timeline
        res[:, 1:1+dim] = path

        if self.attach_volatility:
            res[:, 1+dim:] = v

        return res

    def _generate_heston_process(self, T: float, S0: List[float], time_add=0) -> np.ndarray:
        """
        Generates a sample path with dynamics given by

                                        dS_t = mu S_t dt + sqrt(v_t) S_t dW_t,
        where
                                   dv_t = k(theta - v_t)dt + xi sqrt(v_t) dB_t.

        Here, mu is the drift term of the stock process. k is the mean reversion speed of the volatility process, theta
        the long-run volatility, xi the vol of vol, and W_t, B_t are independent Brownian motions with
        Corr(W_t, B_t) = rho.

        :param T:           Time to simulate till
        :param S0:          Initial stock value
        :param time_add:    Optional. Addition to time index of path
        :return:            Path sim over [0, T] with grid points given by the year mesh in ModelConfig
        """

        return self._heston_generator(T, S0, False, time_add)

    def _generate_rough_heston_process(self, T: float, S0: List[float], time_add=0) -> np.ndarray:
        """
        Generates a sample path with dynamics given by

                                        dS_t = mu S_t dt + sqrt(v_t) S_t dW_t,
        where
                                   dv_t = k(theta - v_t)dt + xi sqrt(v_t) dB_t.

        Here, mu is the drift term of the stock process. k is the mean reversion speed of the volatility process, theta
        the long-run volatility, xi the vol of vol, and W_t, B_t are independent Brownian motions with
        Corr(W_t, B_t) = rho.

        :param T:           Time to simulate till
        :param S0:          Initial stock value
        :param time_add:    Optional. Addition to time index of path
        :return:            Path sim over [0, T] with grid points given by the year mesh in ModelConfig
        """

        return self._heston_generator(T, S0, True, time_add)

    def _heston_generator(self, T: float, S0: List[float], rough: bool, time_add=0):
        """
        Generator class for the Heston model under normal or rough volatility.

        :param T:           Time to simulate till
        :param S0:          Initial stock value
        :param rough:       Boolean. Whether to use rough volatility or not
        :param time_add:    Optional. Addition to time index of path
        :return:
        """

        # Rough parameters
        if rough:
            mu, xi, k, theta, rho, H, V0 = self.params.T
        else:
            mu, xi, k, theta, rho, V0 = self.params.T

        for di in range(self.dim):
            assert 2*k[di]*theta[di] > xi[di]**2, "ERROR: Feller condition not met, dimension {}".format(di+1)

        # Path initialisations
        path_dim    = self.path_dim
        dim         = self.dim
        grid_points = int(self.year_mesh*T) + 1
        dt          = T/(grid_points-1)

        res      = np.zeros((grid_points, path_dim + 1))
        path     = np.zeros((grid_points, dim))
        dB       = np.zeros((grid_points-1, dim))
        dW       = np.zeros((grid_points-1, dim))
        timeline = np.linspace(0, T, grid_points)

        # Deal with S0
        if (len(S0) == path_dim) and self.attach_volatility:
            init_stock = S0[:dim]
            V0         = S0[dim:]
        else:
            init_stock = S0

        # Get bms
        for di in range(dim):
            if rough:
                fractional_driver, brownian_driver = self._generate_rBergomi_drivers(T, H[di], rho[di])
                dB[:, di] = np.diff(fractional_driver)
                dW[:, di] = np.diff(brownian_driver)
            else:
                correlation_matrix = np.array([[1., rho[di]], [rho[di], 1.]])
                bm = self._generate_bm_path(T, correlation_matrix)
                dB[:, di] = np.diff(bm[:, 0])
                dW[:, di] = np.diff(bm[:, 1])

        # Sim volatility first
        v = np.zeros((grid_points, dim))
        path[0, :] = init_stock

        for di in range(dim):
            v[0, di] = V0[di]
            for i in range(grid_points-1):
                if rough:
                    v[i+1, di] = v[i, di] + k[di]*(theta[di] - v[i, di])*dt + xi[di]*dB[i, di]
                else:
                    v[i+1, di] = v[i, di] + k[di]*(theta[di] - v[i, di])*dt + xi[di]*np.sqrt(v[i, di])*dB[i, di]

                path[i + 1, di] = path[i, di]*np.exp((mu[di] - 0.5*v[i, di])*dt + np.sqrt(v[i, di])*dW[i, di])

        # Sim stock path with volatility given
        # if rough:
        #    for di in range(dim):
        #        v[:, di] = np.exp(v[:, di])

        timeline += time_add
        res[:, 0]  = timeline
        res[:, 1:1+dim] = path

        if self.attach_volatility:
            res[:, 1+dim:] = v

        return res

    def _generate_rBergomi_process(self, T: float, S0: List[float], time_add=0) -> np.ndarray:
        """
        Generates an rBergomi process with associated initialised parameters

        :param T:           Time to simulate till
        :param S0:          Initial stock value
        :param time_add:    Optional. Addition to time index of path
        :return:            Path sim over [0, T] with grid points given by the year mesh in ModelConfig
        """

        xi0, nu, rho, H = self.params.T

        grid_points = int(self.year_mesh*T) + 1
        dt          = T/(grid_points-1)
        path_dim    = self.path_dim
        dim         = self.dim

        # Generate path objects
        res      = np.zeros((grid_points, path_dim + 1))
        timeline = np.linspace(0, T, grid_points)
        path     = np.zeros((grid_points, dim))
        v        = np.zeros((grid_points, dim))

        if (len(S0) == path_dim) and self.attach_volatility:
            init_stock = S0[:dim]
            xi0        = S0[dim:]
        else:
            init_stock = S0

        # Generate required number of drivers, generate each path
        for di in range(dim):
            # Set parameters
            xi0_i, nu_i, rho_i, H_i = xi0[di], nu[di], rho[di], H[di]

            # Generate rBergomi drivers
            fbm, bm = self._generate_rBergomi_drivers(T, H_i, rho_i)

            # Get values and variance process
            C_H = np.power(2*H_i*gamma(3.0 / 2.0 - H_i)/gamma(H_i+0.5)/gamma(2.0 - 2.0 * H_i), 0.5)
            variance = xi0_i*np.exp(2*nu_i*C_H*fbm - np.power(nu_i*C_H, 2) * np.power(timeline, 2*H_i) / H_i)
            v[:, di] = variance

            # Initialise log stock process
            log_stock = np.ones(grid_points)
            log_stock[0] = np.log(init_stock[di])

            # Get Brownian driver
            dw = np.diff(bm)

            # Build path
            for i in range(grid_points - 1):
                log_stock[i + 1] = log_stock[i] - 0.5*variance[i]*dt + np.power(variance[i], 0.5)*dw[i]

            path[:, di] = np.exp(log_stock)

        timeline += time_add
        res[:, 0]       = timeline
        res[:, 1:1+dim] = path

        if self.attach_volatility:
            res[:, 1+dim:] = v

        return res

    def _generate_bm_path(self, T: float, cov: np.ndarray) -> np.ndarray:
        """
        Generates a d-dimensional Brownian motion with correlation matrix specified by :param corr:.

        :param T:       Time window (in years) to simulate over
        :param cov:     Covariance matrix
        :return:        d-dimensional Brownian motion
        """

        d = cov.shape[0]

        grid_points = int(self.year_mesh*T) + 1
        dt = np.sqrt(T/(grid_points-1))

        bm = np.zeros((grid_points, d))
        dw = np.random.multivariate_normal(mean=[0.]*d, cov=cov, size=grid_points-1)
        bm[1:, :] = dt*np.cumsum(dw, axis=0)

        return bm

    def _generate_rBergomi_drivers(self, T: float, H: float, rho: float) -> np.ndarray:
        """
        Generates rBergomi drivers, including the fBm and bm paths.

        :param T:           Time to simulate til
        :param H:           Hurst exponent in Riemann-Louiville kernel
        :param rho:         Correlation between driving bms
        :return:            fBm and correlated bm driver
        """

        grid_points = int(self.year_mesh*T) + 1
        dt = T/(grid_points - 1)

        # Generate bm driver
        dZ1 = np.random.normal(loc=0, scale=1, size=grid_points)
        dZ2 = np.random.normal(loc=0, scale=1, size=grid_points)
        dW = np.sqrt(dt)*(rho*dZ1 + np.sqrt(1-rho*rho)*dZ2)

        i = np.arange(grid_points - 1) + 1
        opt_k = np.power((np.power(i, 2*H) - np.power(i-1., 2*H)) / (2.0*H), 0.5)*np.power(dt, H-0.5)

        # Convolution to get fBm
        Y = np.zeros(grid_points)
        Y[1:] = np.convolve(opt_k, dW)[:grid_points-1]

        # Return correlated BM
        dB = np.sqrt(dt)*dZ1

        return np.array([Y, dB.cumsum()])

    def _generate_fbm_process(self, T: float, S0: List[float], time_add=0):
        """
        Generates a fBm process with associated initialised parameters

        :param T:           Time to simulate till
        :param S0:          Initial stock value
        :param time_add:    Optional. Addition to time index of path
        :return:            Path sim over [0, T] with grid points given by the year mesh in ModelConfig
        """

        H, beta = self.params.T
        dim     = self.dim

        grid_points = int(self.year_mesh*T)
        ii = np.linspace(0, grid_points-1, grid_points, dtype=np.int16) + 1
        dt = T/(grid_points - 1)

        res = np.zeros((grid_points, dim+1))
        path = np.zeros((grid_points, dim))

        for di in range(dim):
            if beta[di] == 0.0:
                opt_k = np.power((np.power(ii, 2*H[di]) - np.power(ii-1., 2*H[di])) / (2.0*H[di]), 0.5)*np.power(dt, H[di]-0.5)
            else:
                opt_k = 1/np.sqrt(dt)*np.sqrt(
                    np.power(4*beta[di]**2, -H[di])*gamma(2*H[di])*(gammainc(2*H[di], 2*beta[di]*dt*ii) - gammainc(2*H[di], 2*beta[di]*dt*(ii - 1)))
                )

            dW = np.random.normal(0, np.sqrt(dt), size=grid_points-1)

            path[1:, di] = S0[di]*np.convolve(opt_k, dW)[:grid_points - 1]

        ii -=1
        ii += time_add
        res[:, 0] = ii
        res[:, 1:1 + dim] = path

        return res
