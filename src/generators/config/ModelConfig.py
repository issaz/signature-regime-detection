from utils.Config import Config


_default_args = {
    "year_mesh": 7*252,
    "models": [
        'gbm'         ,
        'merton'      ,
        'heston'      ,
        'rough_heston',
        'rBergomi'    ,
        'ar_p'        ,
        'i_ar_p'      ,
        'fbm'
    ],
    "model_param_length": Config(**{
        "gbm"         : 2,
        "merton"      : 5,
        'heston'      : 6,
        'rough_heston': 7,
        'rBergomi'    : 4,
        'ar_p'        : 2,
        'i_ar_p'      : 2,
        'fbm'         : 2
    }),
    "model_param_names" : Config(**{
        "gbm"         : ['mu', 'sigma'],
        "merton"      : ['mu', 'sigma', 'lambda', 'muJ', 'sigmaJ'],
        'heston'      : ['mu', 'xi', 'k', 'theta', 'rho', 'V0'],
        'rough_heston': ['mu', 'xi', 'k', 'theta', 'rho', 'H', 'V0'],
        'rBergomi'    : ['xi0', 'nu', 'rho', 'H'],
        'ar_p'        : ['roots', 'sd_n'],
        'i_arp'       : ['roots', 'sd_n'],
        'fbm'         : ['H', 'beta']
    }),
    "attach_volatility": False
}


class ModelConfig(Config):
    """@DynamicAttrs"""

    def __init__(self):
        super().__init__(**_default_args)
