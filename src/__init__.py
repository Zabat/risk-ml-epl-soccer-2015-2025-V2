"""
EPL Betting Model
=================

A machine learning system for predicting English Premier League home wins.

Modules:
    predictor: Main prediction model (Logistic Regression)
    backtest: Backtesting framework
    risk_analysis: Risk metrics and analysis
    utils: Utility functions

Example:
    >>> from src import EPLHomeWinPredictor, run_backtest
    >>> predictor = EPLHomeWinPredictor()
    >>> predictor.fit(training_data)
    >>> result = predictor.predict_match(home_elo=1900, away_elo=1700)

Author: Your Name
License: MIT
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'Your Name'
__license__ = 'MIT'

# Main classes
from .predictor import EPLHomeWinPredictor, load_model, quick_predict
from .backtest import BettingBacktest, BacktestConfig, run_backtest, print_performance_report

# Risk analysis functions
from .risk_analysis import (
    kelly_fraction,
    fractional_kelly,
    calculate_var,
    calculate_cvar,
    optimal_leverage_for_capital,
    run_monte_carlo,
    analyze_kelly_variants
)

# Utility functions
from .utils import (
    load_epl_data,
    validate_data,
    train_test_split_temporal,
    setup_plot_style,
    plot_bankroll_evolution,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)

__all__ = [
    # Classes
    'EPLHomeWinPredictor',
    'BettingBacktest',
    'BacktestConfig',
    
    # Main functions
    'run_backtest',
    'load_model',
    'quick_predict',
    'print_performance_report',
    
    # Risk analysis
    'kelly_fraction',
    'fractional_kelly',
    'calculate_var',
    'calculate_cvar',
    'optimal_leverage_for_capital',
    'run_monte_carlo',
    'analyze_kelly_variants',
    
    # Utilities
    'load_epl_data',
    'validate_data',
    'train_test_split_temporal',
    'setup_plot_style',
    'plot_bankroll_evolution',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown'
]
