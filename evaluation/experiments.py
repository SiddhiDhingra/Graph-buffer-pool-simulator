"""Legacy entry point redirected to THGB-only experiments.

Use ``python run_research.py`` or ``python -m evaluation.run_thgb_experiments``.
"""
from evaluation.run_thgb_experiments import main

__all__ = ["main"]
