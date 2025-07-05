"""
TransAct: Transformer-based Realtime User Action Model for Recommendation

This package implements the TransAct model from Pinterest's research paper
"TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest" (KDD '23).
"""

from .transact import TransActModel
from .data import UserActionDataset, UserActionDataLoader
from .training import TransActTrainer
from .evaluation import TransActEvaluator

__version__ = "0.1.0"
__all__ = [
    "TransActModel",
    "UserActionDataset", 
    "UserActionDataLoader",
    "TransActTrainer",
    "TransActEvaluator"
] 