"""Base feature type."""


class Feature:
    """Lightweight feature base class."""

    visibility = ("owner",)

    def vector(self):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def set_values(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
