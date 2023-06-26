from src.utils.Config import Config


_default_args = {
    "transformations": {
        "standardise_path_transform"  : (True, 0, {"s_type": "initial"}),
        "time_normalisation_transform": (False, 0, {}),
        "time_difference_transform"   : (False, 0, {}),
        "difference_transform"        : (False, 0, {}),
        "translation_transform"       : (True, 0, {"all_channels": True}),
        "scaling_transform"           : (False, 0, {"sigmas": 1.}),
        "cumulant_transform"          : (False, 2, {}),
        "increment_transform"         : (True, 2, {}),
        "lead_lag_transform"          : (False, 3, {}),
        "invisibility_transform"      : (False, 4, {}),
    },
    "compute_pathwise_signature_transform": False,
    "signature_order": 4
}


# noinspection PyAttributeOutsideInit
class PathTransformerConfig(Config):
    """@DynamicAttrs"""

    def __init__(self):
        super().__init__(**_default_args)

    def set_transformations(self, transformations_dict: dict):
        self.transformations = transformations_dict

