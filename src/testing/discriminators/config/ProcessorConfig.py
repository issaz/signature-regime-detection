from src.utils.Config import Config


_default_args = {
    "generalmmddetector_kwargs": Config(**{
        "n_tests": 1000,
        "n_evaluations": 1,
        "metric_kwargs": Config(**{
            "kernel_type": "rbf",
            "metric_type": "mmd",
            "sigmas": [1e0],
            "dyadic_orders": [2],
            "lambd": 5
        }),
        "evaluator_kwargs": Config(**{
            "pct_ignore": 0.2
        })
    }),
    "truncatedmmddetector_kwargs": Config(**{
        "n_tests": 1000,
        "n_evaluations": 1,
        "metric_kwargs": Config(**{
            "signature_order": 3,
            "scale_signature": False,
            "sigma": 1
        })
    }),
    "anomalydetector_kwargs": Config(**{
        "signature_depth": 6,
        "signature_type": "signature",
        "pct_path_bank": 0.01
    }),
    "autoevaluator_kwargs": Config(**{
        "metric_kwargs": Config(**{
            "kernel_type": "rbf",
            "metric_type": "mmd",
            "sigmas": [0.0025],
            "dyadic_orders": [3],
            "lambd": 5
        }),
        "n_scores": 100,
        "evaluator_kwargs": Config(**{
            "lags": [-1],
            "threshold_method": "bootstrap"
        })
    }),
    "truncatedautoevaluator_kwargs": Config(**{
        "metric_kwargs": Config(**{
            "signature_order": 3,
            "scale_signature": False,
            "sigma": 1
        }),
        "n_scores": 100,
        "evaluator_kwargs": Config(**{
            "lags": [-1]
        })
    }),
    "alpha_value": 0.95,
    "device": "cuda:0",
    "overwrite_prior": False,
    "jobs": 4
}


class ProcessorConfig(Config):
    """@DynamicAttrs"""

    def __init__(self):
        super().__init__(**_default_args)
