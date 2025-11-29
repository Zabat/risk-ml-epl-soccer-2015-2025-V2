"""
Utility Functions
=================

Helper functions for data loading, preprocessing, and visualization.

Author: Your Name
License: MIT
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# DATA LOADING
# ============================================================

def load_epl_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load and preprocess EPL match data.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(filepath)
    
    # Parse dates
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    
    # Sort chronologically
    df = df.sort_values('MatchDate').reset_index(drop=True)
    
    # Add derived columns
    df['TotalGoals'] = df['FTHome'] + df['FTAway']
    df['GoalDiff'] = df['FTHome'] - df['FTAway']
    
    return df


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, list of missing columns)
    """
    required = [
        'HomeTeam', 'AwayTeam', 'HomeElo', 'AwayElo',
        'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
        'FTResult', 'OddHome'
    ]
    
    missing = [col for col in required if col not in df.columns]
    is_valid = len(missing) == 0
    
    return is_valid, missing


def train_test_split_temporal(
    df: pd.DataFrame,
    test_size: float = 0.2,
    date_col: str = 'MatchDate'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (for time series).
    
    Args:
        df: DataFrame to split
        test_size: Fraction of data to use for testing
        date_col: Column containing dates
        
    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values(date_col)
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


# ============================================================
# VISUALIZATION
# ============================================================

# Color scheme
COLORS = {
    'primary': '#3498db',
    'success': '#2ecc71',
    'danger': '#e74c3c',
    'warning': '#f39c12',
    'info': '#9b59b6',
    'dark': '#2c3e50',
    'light': '#ecf0f1'
}


def setup_plot_style():
    """Configure matplotlib for consistent styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11


def plot_bankroll_evolution(
    bankroll_history: List[float],
    initial_bankroll: float = 1000,
    title: str = 'Bankroll Evolution',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bankroll evolution over time.
    
    Args:
        bankroll_history: List of bankroll values
        initial_bankroll: Starting capital
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Main line
    ax.plot(bankroll_history, color=COLORS['primary'], linewidth=2, label='Bankroll')
    
    # Initial capital reference
    ax.axhline(initial_bankroll, color=COLORS['danger'], linestyle='--', 
               alpha=0.7, label=f'Initial ({initial_bankroll}€)')
    
    # Fill profit/loss area
    final_bankroll = bankroll_history[-1]
    fill_color = COLORS['success'] if final_bankroll >= initial_bankroll else COLORS['danger']
    ax.fill_between(range(len(bankroll_history)), initial_bankroll, bankroll_history,
                    alpha=0.3, color=fill_color)
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Bankroll (€)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = 'Model Calibration',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot calibration curve.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate calibration
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    
    bin_probs = []
    bin_freqs = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_probs.append(y_prob[mask].mean())
            bin_freqs.append(y_true[mask].mean())
    
    # Plot
    ax.scatter(bin_probs, bin_freqs, s=100, c=COLORS['primary'], 
               zorder=3, edgecolors='white', linewidth=2)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration', linewidth=2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Frequency')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def plot_profit_distribution(
    profits: np.ndarray,
    title: str = 'Profit Distribution',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of profits/losses.
    
    Args:
        profits: Array of profit values
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gains = profits[profits > 0]
    losses = profits[profits < 0]
    
    ax.hist(gains, bins=20, alpha=0.6, color=COLORS['success'], label='Gains')
    ax.hist(losses, bins=20, alpha=0.6, color=COLORS['danger'], label='Losses')
    ax.axvline(profits.mean(), color=COLORS['primary'], linestyle='--',
               linewidth=2, label=f'Mean: {profits.mean():.1f}€')
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Profit (€)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


# ============================================================
# FORMATTING
# ============================================================

def format_currency(value: float, currency: str = '€') -> str:
    """Format number as currency."""
    return f"{value:,.0f}{currency}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format number as percentage."""
    return f"{value*100:,.{decimals}f}%"


def format_odds(value: float, decimals: int = 2) -> str:
    """Format decimal odds."""
    return f"{value:.{decimals}f}"


# ============================================================
# STATISTICS
# ============================================================

def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Array of periodic returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(periods_per_year)


def calculate_max_drawdown(values: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.
    
    Args:
        values: Array of portfolio values
        
    Returns:
        Tuple of (max_drawdown, peak_index, trough_index)
    """
    running_max = np.maximum.accumulate(values)
    drawdowns = (running_max - values) / running_max
    
    max_dd = np.max(drawdowns)
    trough_idx = np.argmax(drawdowns)
    peak_idx = np.argmax(values[:trough_idx + 1])
    
    return max_dd, peak_idx, trough_idx


def calculate_sortino_ratio(
    returns: np.ndarray,
    target_return: float = 0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (downside deviation-adjusted return).
    
    Args:
        returns: Array of periodic returns
        target_return: Target return threshold
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - target_return / periods_per_year
    downside_returns = np.minimum(excess_returns, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return 0.0
    
    return (np.mean(excess_returns) / downside_deviation) * np.sqrt(periods_per_year)


if __name__ == "__main__":
    print("Utility Functions Module")
    print("="*50)
    
    # Demo: Sharpe ratio
    returns = np.random.normal(0.001, 0.02, 252)
    sharpe = calculate_sharpe_ratio(returns)
    print(f"Demo Sharpe Ratio: {sharpe:.2f}")
