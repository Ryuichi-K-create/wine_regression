# src/__init__.py
"""
Wine Quality Regression Project

A machine learning project for predicting wine quality using neural networks.
"""

__version__ = "1.0.0"

from .config import Config
from .utils import (
    load_dataframe,
    compute_regression_metrics,
    set_seed,
    make_run_dir
)
from .models import MLPRegressor, LinearRegressor

__all__ = [
    'Config',
    'load_dataframe', 
    'compute_regression_metrics',
    'set_seed',
    'make_run_dir',
    'MLPRegressor',
    'LinearRegressor'
]