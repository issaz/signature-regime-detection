from src.utils.Config import Config


_default_args = {
    "n_regime_changes": 10,
    "f_length_scale"  : 0.5,
    "type"            : "random_on_off_steps",  # random_start_fixed_length
    "r_on_args"       : ["poisson", 2],
    "r_off_args"      : ["poisson", 1/49],
    "r_min_distance"  : 10,
    "r_min_gap"       : 10
}


class RegimePartitionerConfig(Config):
    """@DynamicAttrs"""

    def __init__(self):
        super().__init__(**_default_args)
