"""Registry of the routing techniques this supplementary material ships.

The paper's experiments fit only these two, and the scripts here import
nothing else.
"""
from .aggregator import AggregatorRouter
from .jury import JuryRouter

REGISTRY = {cls.name: cls for cls in [AggregatorRouter, JuryRouter]}

__all__ = ["REGISTRY"]
