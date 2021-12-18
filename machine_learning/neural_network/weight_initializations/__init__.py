"""Load weight initialization functions and pass them to the outside world."""

__all__ = [
    "xavier",
    "he",
]

from .he import he
from .xavier import xavier
