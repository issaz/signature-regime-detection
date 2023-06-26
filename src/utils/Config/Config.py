assertMsg = "New keys not subset of default keys"


class Config(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def override_args(self, overrides: dict):
        assert set(overrides.keys()).issubset(set(self.__dict__.keys())), assertMsg
        for k, v in overrides.items():
            setattr(self, k, v)

#    def __repr__(self):

        """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                parts.append(f"{k}:")
                parts.append(v.__repr__())
            elif v is None:
                continue
            else:
                val = ", ".join(v) if isinstance(v, list) else v
                parts.append("{}: {}".format(k, val))
        return '\n'.join(parts)
        """
