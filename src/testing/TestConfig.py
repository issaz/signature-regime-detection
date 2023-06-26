from src.utils.Config import Config


_default_args = {
    "n_steps"                       : 7,
    "n_paths"                       : 10,
    "weight_factor"                 : 1,
    "offset"                        : 0,
    "belief_models"                 : ["gbm"],
    "model_pair_names"              : ["gbm", "gbm"],
    "belief_params"                 : [[[0., 0.2] for _ in range(1)]],
    "model_pair_params"             : [[[0., 0.2] for _ in range(1)], [[0., 0.3] for _ in range(1)]],
    "path_bank_size"                : 100000,
    "n_runs"                        : 10,
    "time_sim"                      : 4,
    "detector"                      : "GeneralMMDDetector",
    "generalmmddetector_kwargs"     : Config(**{"eval_kwargs": {"evaluation": "total"}}),
    "anomalydetector_kwargs"        : Config(**{"eval_kwargs": {}}),
    "truncatedmmddetector_kwargs"   : Config(**{"eval_kwargs": {"evaluation": "total"}})
}


class TestConfig(Config):
    """@DynamicAttrs"""

    def __init__(self):
        super().__init__(**_default_args)
