from typing import List

import numpy as np

from src.utils.auxiliary_classes.RegimePartitioner import RegimePartitionerConfig
from src.generators.Model import Model
from src.utils.helper_functions.global_helper_functions import roundrobin


class RegimePartitioner(object):
    """
    Object which breaks an interval into regimes and builds regime change path.
    """

    def __init__(self, year_mesh: int, config: RegimePartitionerConfig):
        """
        Initialises object with corresponding year mesh and config object.

        :param year_mesh:   Int, how fine to partition one year into a mesh grid
        :param config:      RegimePartitionerConfig object
        """

        self.n_regime_changes = config.n_regime_changes
        self.f_length_scale   = config.f_length_scale
        self.type             = config.type
        self.r_on_args        = config.r_on_args
        self.r_off_args       = config.r_off_args
        self.r_min_distance   = config.r_min_distance
        self.r_min_gap        = config.r_min_gap

        self.year_mesh = year_mesh

        self.normal_regimes = None
        self.regime_changes = None
        self.time_indexes   = None

        if self.type == "random_on_off_steps":
            self.generate_regime_partitions = self._generate_regime_partitions_on_off
        else:
            self.generate_regime_partitions = self._generate_regime_partitions_fixed_length

    def _generate_regime_partitions_on_off(self, T: float, n_steps: int):
        """
        Generates regime partition according to two-step switching/staying model (as standard, two Poisson random
        variables)

        :param T:           Time to generate regime partitions till
        :param n_steps:     Way to break up partitioned block into chunks of sub-paths
        :return:            Normal regimes, regime changes, and time indexes
        """

        year_mesh           = self.year_mesh
        on_args             = (self.r_on_args[1]*n_steps)/year_mesh
        off_args            = self.r_off_args[1]
        min_regime_distance = self.r_min_distance
        min_regime_gap      = self.r_min_gap

        normal_regimes = []
        regime_changes = []

        on_variable   = getattr(np.random, self.r_on_args[0])
        off_variable  = getattr(np.random, self.r_off_args[0])

        last_index = 0
        regime_off_flag = True
        time_regime_elapsed = 0
        time_since_regime   = 0

        for i in range(0, int(T * year_mesh), n_steps):
            if regime_off_flag:
                # Check if we enter new regime
                regime_flag = on_variable(on_args)
                if (regime_flag > 0) and (time_since_regime >= min_regime_gap):
                    # Regime change
                    if i > 0:
                        normal_regimes.append([last_index, i - 1])
                    last_index = i
                    regime_off_flag = False
                    time_since_regime = 0
                else:
                    time_since_regime += 1
            else:
                # Check if we exit the regime
                regime_out = off_variable(off_args)
                if (regime_out > 0) and (time_regime_elapsed >= min_regime_distance):
                    # Regime change has ended
                    regime_changes.append([last_index, i - 1])
                    last_index = i
                    regime_off_flag = True
                    time_regime_elapsed = 0
                time_regime_elapsed += 1

        # Check how we ended
        final_bracket = [last_index, int(T * year_mesh)]
        if regime_off_flag:
            normal_regimes.append(final_bracket)
        else:
            regime_changes.append(final_bracket)

        time_indexes = list(roundrobin(normal_regimes, regime_changes))

        self.normal_regimes, self.regime_changes, self.time_indexes = normal_regimes, regime_changes, time_indexes

    def _generate_regime_partitions_fixed_length(self, T: float):
        """
        Generates regime partitions over [0, T] with the corresponding configuration as initialised

        :param T:   Final time to simulate over, in years
        :return:    None. Assigns value to class objects: divisions of mesh grid into regime and non-regime periods
        """
        n_regime_changes = self.n_regime_changes
        regime_changes = []

        while len(regime_changes) != n_regime_changes:

            regime_lengths = [int(self.year_mesh*self.f_length_scale) for _ in range(n_regime_changes)]

            regime_starts = sorted([np.random.randint(0, (T-1)*self.year_mesh) for _ in range(n_regime_changes)])
            regime_ends = [s+l for s, l in zip(regime_starts, regime_lengths)]

            # Specify which grid points are changes in the regime
            regime_changes = [[s, e] for s, e in zip(regime_starts, regime_ends)]

            # Filter out overlaps
            regime_changes = [
                 item for item, t in zip(regime_changes[:-1], regime_changes[1:]) if item[1]+2 < t[0]
            ] + [regime_changes[-1]]

        # Fill the rest as the "normal" period
        normal_regimes = [[0, regime_changes[0][0] - 1]] + \
                         [[r1[1] + 1, r2[0] - 1] for r1, r2 in zip(regime_changes[:-1], regime_changes[1:])] + \
                         [[regime_changes[-1][1] + 1, self.year_mesh * T]]

        # Splice all the time indexes together
        time_indexes = list(roundrobin(normal_regimes, regime_changes))

        self.normal_regimes, self.regime_changes, self.time_indexes = normal_regimes, regime_changes, time_indexes

    def generate_regime_change_path(self, models: List[Model], S0: List[float]) -> np.ndarray:
        """
        Generates a regime-changed path corresponding to regime divisions from previous function, given Model objects

        :param models:      List of model pairs to cycle over
        :param S0:          Initial stock value
        :return:            Regime-changed path at appropriate times
        """

        assert len(set((m.dim for m in models))) == 1, "Error: Models have different dimensions."

        vol_inc = all((m.attach_volatility for m in models))
        p_dim = models[0].path_dim
        n_models = len(models)

        if vol_inc and (len(S0) != p_dim):

            def get_index(model_type: str):
                if model_type in ["heston", "gbm"]:
                    return -1
                else:
                    return 0

            init_S0 = S0 + [p[get_index(models[0].model_type)] for p in models[0].params]
        else:
            init_S0 = S0

        path = np.array([[0.] + init_S0])

        for i, ind in enumerate(self.time_indexes):
            # Get fraction of time to simulate
            dt = 1/self.year_mesh
            frac_t = np.diff(ind)[0]*dt

            if i != 0:
                frac_t += dt

            # Get starting path value
            # this_S0 = path[-1, 1:m_dim+1] if vol_inc else path[-1, 1:]
            this_S0 = path[-1, 1:]

            # Get model index
            index = i % n_models
            tp    = models[index].sim_path(T=frac_t, S0=this_S0, time_add=path[-1, 0])

            path = np.concatenate(
                [path, tp[1:]],
                axis=0
            )

        return path

    def changes_to_times(self):
        """
        Turns list of list into arrays divided by year_mesh

        :return: If generated, regime changes and so on as arrays. Else none.
        """

        if self.normal_regimes is None:
            return None, None, None
        else:
            def adw(x):
                return np.array(x)/self.year_mesh

            return adw(self.normal_regimes), adw(self.regime_changes), adw(self.time_indexes)
