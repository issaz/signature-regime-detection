from src.utils.Config import Config


_default_args = {
    "n_clusters": 2,
    "hierarchicalclusterer_kwargs": Config(**{
        "linkage": "average"  # Can be ["single", "average", "complete"]
    }),
    "wassersteinkmeans_kwargs": Config(**{
        "window_length": 70,
        "overlap"      : 63,
        "wp"           : 2
    })
}


class ClusterConfig(Config):
    """@DynamicAttrs"""

    def __init__(self):
        super().__init__(**_default_args)
