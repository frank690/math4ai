"""Load cost functions and pass them to the outside world."""

__all__ = [
    "cross_entropy_loss",
    "hinge_loss",
]

from .cross_entropy import cross_entropy_loss
from .hinge import hinge_loss
