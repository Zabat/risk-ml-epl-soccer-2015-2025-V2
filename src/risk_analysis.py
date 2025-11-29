"""
Risk Analysis Module
====================

Advanced risk analysis tools including:
- Kelly Criterion variants
- CVaR (Conditional Value at Risk)
- Monte Carlo simulations
- Dynamic leverage optimization
- Stress testing

Example:
    >>> from src.risk_analysis import run_monte_carlo, calculate_cvar
    >>> results = run_monte_carlo(bet_opportunities, n_simulations=1000)
    >>> cvar = calculate_cvar(results, confidence=0.95)

Author: Your Name
License: MIT
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def kelly_fraction(prob: float, odds: float) -> float:
    """
    Calculate full Kelly fraction.
    
    The Kelly Criterion maximizes long-term growth rate:
    f* = (p * b - q) / b
    
    Args:
        prob: Probability of winning (0 to 1)
        odds: Decimal odds
        
    Returns:
        Optimal bet fraction (0 to 1)
    """
    if odds <= 1 or prob <= 0 or prob >= 1:
        return 0.0
    
    q = 1 - prob
    b = odds - 1
    kelly = (prob * b - q) / b
    
    return max(0.0, kelly)


def fractional_kelly(prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Calculate fractional Kelly bet size.
    
    Fractional Kelly reduces variance at the cost of some expected growth.
    Common choices:
    - 1.00: Full Kelly (maximum growth, high variance)
    - 0.50: Half Kelly (good balance)
    - 0.25: Quarter Kelly (conservative, low variance)
    
    Args:
        prob: Win probability
        odds: Decimal odds
        fraction: Kelly fraction (0 to 1)
        
    Returns:
        Recommended bet fraction
    """
    full_kelly = kelly_fraction(prob, odds)
    return full_kelly * fraction


def calculate_var(
    returns: np.ndarray,
    confidence: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    VaR is the threshold value such that the probability of
    loss exceeding this value is (1 - confidence).
    
    Args:
        returns: Array of returns or final values
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        VaR value (returns at the (1-confidence) percentile)
    """
    return np.percentile(returns, (1 - confidence) * 100)


def calculate_cvar(
    returns: np.ndarray,
    confidence: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
    
    CVaR is the expected loss given that a loss exceeds VaR.
    It's a more conservative measure that accounts for tail risk.
    
    CVaR = E[X | X <= VaR]
    
    Args:
        returns: Array of returns or final values
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        CVaR value (expected value in worst (1-confidence) cases)
    """
    var_threshold = calculate_var(returns, confidence)
    tail_values = returns[returns <= var_threshold]
    
    if len(tail_values) == 0:
        return var_threshold
    
    return float(np.mean(tail_values))


def optimal_leverage_for_capital(
    capital: float,
    initial_capital: float,
    max_leverage: float = 3.0,
    risk_tolerance: float = 0.5,
    critical_threshold: Optional[float] = None
) -> float:
    """
    Calculate optimal leverage as a function of current capital.
    
    Strategy:
    - Use high leverage when capital is low (aggressive recovery)
    - Reduce leverage as capital grows (protect gains)
    - This mimics professional fund management practices
    
    Args:
        capital: Current capital
        initial_capital: Starting capital
        max_leverage: Maximum leverage to use
        risk_tolerance: Aggressiveness factor (0 to 1)
        critical_threshold: Capital level above which to reduce leverage
        
    Returns:
        Optimal leverage multiplier
    """
    if critical_threshold is None:
        critical_threshold = initial_capital * 1.5
    
    capital_ratio = capital / initial_capital
    
    if capital <= initial_capital * 0.5:
        # Critical zone: maximum leverage for recovery
        return max_leverage * risk_tolerance
    
    elif capital <= initial_capital:
        # Recovery zone: high leverage
        t = (capital - initial_capital * 0.5) / (initial_capital * 0.5)
        return max_leverage * (1 - 0.3 * t)
    
    elif capital <= critical_threshold:
        # Growth zone: moderate leverage
        t = (capital - initial_capital) / (critical_threshold - initial_capital)
        return max_leverage * (0.7 - 0.3 * t)
    
    else:
        # Protection zone: low leverage
        excess_ratio = capital / critical_threshold
        return max_leverage * 0.4 / np.sqrt(excess_ratio)


def run_monte_carlo(
    bet_opportunities: List[Dict],
    n_simulations: int = 1000,
    initial_bankroll: float = 1000,
    kelly_fraction: float = 0.25,
    max_bet: float = 0.10,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Run Monte Carlo simulation of betting strategy.
    
    Simulates multiple possible outcomes based on predicted probabilities
    to estimate the distribution of final bankroll values.
    
    Args:
        bet_opportunities: List of dicts with 'prob' and 'odds' keys
        n_simulations: Number of simulations to run
        initial_bankroll: Starting capital
        kelly_fraction: Kelly fraction to use
        max_bet: Maximum bet as fraction of bankroll
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with simulation results:
            - final_bankrolls: Array of final values
            - mean, median, std
            - var_95, var_99
            - cvar_95, cvar_99
            - prob_profit, prob_ruin
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    final_bankrolls = []
    
    for sim in range(n_simulations):
        bankroll = initial_bankroll
        
        for bet in bet_opportunities:
            prob = bet['prob']
            odds = bet['odds']
            
            # Calculate bet size
            full_kelly_frac = kelly_fraction(prob, odds)
            bet_frac = min(full_kelly_frac * kelly_fraction, max_bet)
            
            if bet_frac > 0.005:
                bet_amount = bankroll * bet_frac
                
                # Simulate outcome
                if np.random.random() < prob:
                    bankroll += bet_amount * (odds - 1)
                else:
                    bankroll -= bet_amount
            
            # Stop if ruined
            if bankroll < 10:
                bankroll = 0
                break
        
        final_bankrolls.append(bankroll)
    
    finals = np.array(final_bankrolls)
    
    return {
        'final_bankrolls': finals,
        'mean': float(np.mean(finals)),
        'median': float(np.median(finals)),
        'std': float(np.std(finals)),
        'var_95': float(calculate_var(finals, 0.95)),
        'var_99': float(calculate_var(finals, 0.99)),
        'cvar_95': float(calculate_cvar(finals, 0.95)),
        'cvar_99': float(calculate_cvar(finals, 0.99)),
        'prob_profit': float((finals > initial_bankroll).mean()),
        'prob_ruin': float((finals < initial_bankroll * 0.1).mean()),
        'max_final': float(np.max(finals)),
        'min_final': float(np.min(finals)),
        'n_simulations': n_simulations
    }


def stress_test_consecutive_losses(
    bet_opportunities: List[Dict],
    n_consecutive_losses: int,
    initial_bankroll: float = 1000,
    kelly_fraction_val: float = 0.25,
    max_bet: float = 0.10
) -> Dict:
    """
    Stress test: What happens with N consecutive losses at start?
    
    Args:
        bet_opportunities: List of betting opportunities
        n_consecutive_losses: Number of forced losses at start
        initial_bankroll: Starting capital
        kelly_fraction_val: Kelly fraction to use
        max_bet: Maximum bet size
        
    Returns:
        Dictionary with stress test results
    """
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    forced_losses = 0
    
    for bet in bet_opportunities:
        prob = bet['prob']
        odds = bet['odds']
        
        full_kelly = kelly_fraction(prob, odds)
        bet_frac = min(full_kelly * kelly_fraction_val, max_bet)
        
        if bet_frac > 0.005:
            bet_amount = bankroll * bet_frac
            
            # Force losses for first N bets
            if forced_losses < n_consecutive_losses:
                bankroll -= bet_amount
                forced_losses += 1
            else:
                # Use actual probability
                if np.random.random() < prob:
                    bankroll += bet_amount * (odds - 1)
                else:
                    bankroll -= bet_amount
        
        bankroll = max(0, bankroll)
        bankroll_history.append(bankroll)
        
        if bankroll < 1:
            break
    
    return {
        'n_forced_losses': n_consecutive_losses,
        'final_bankroll': bankroll,
        'min_bankroll': min(bankroll_history),
        'recovered': bankroll > initial_bankroll,
        'bankroll_history': bankroll_history
    }


def analyze_kelly_variants(
    bet_opportunities: List[Dict],
    initial_bankroll: float = 1000,
    variants: Optional[List[Tuple[str, float]]] = None
) -> pd.DataFrame:
    """
    Compare different Kelly criterion variants.
    
    Args:
        bet_opportunities: List of betting opportunities
        initial_bankroll: Starting capital
        variants: List of (name, fraction) tuples to test
        
    Returns:
        DataFrame with results for each variant
    """
    if variants is None:
        variants = [
            ('Full Kelly', 1.0),
            ('3/4 Kelly', 0.75),
            ('Half Kelly', 0.5),
            ('Quarter Kelly', 0.25),
            ('Eighth Kelly', 0.125)
        ]
    
    results = []
    
    for name, frac in variants:
        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        peak = initial_bankroll
        max_drawdown = 0
        
        for bet in bet_opportunities:
            prob = bet['prob']
            odds = bet['odds']
            actual_won = bet.get('actual_result', np.random.random() < prob)
            
            full_kelly = kelly_fraction(prob, odds)
            bet_frac = min(full_kelly * frac, 0.15)
            
            if bet_frac > 0.005:
                bet_amount = bankroll * bet_frac
                
                if actual_won:
                    bankroll += bet_amount * (odds - 1)
                else:
                    bankroll -= bet_amount
            
            bankroll = max(0, bankroll)
            bankroll_history.append(bankroll)
            
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, dd)
        
        # Calculate Sharpe ratio
        returns = np.diff(bankroll_history) / np.array(bankroll_history[:-1])
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        results.append({
            'variant': name,
            'kelly_fraction': frac,
            'final_bankroll': bankroll,
            'return_pct': (bankroll / initial_bankroll - 1) * 100,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe
        })
    
    return pd.DataFrame(results)


def calculate_optimal_kelly(
    historical_results: pd.DataFrame,
    prob_col: str = 'prob_predicted',
    odds_col: str = 'odds',
    result_col: str = 'result'
) -> Dict:
    """
    Estimate optimal Kelly fraction from historical results.
    
    Uses historical win rate to adjust Kelly recommendations.
    
    Args:
        historical_results: DataFrame with bet history
        prob_col: Column with predicted probabilities
        odds_col: Column with odds
        result_col: Column with results ('WIN' or 'LOSS')
        
    Returns:
        Dictionary with optimal Kelly analysis
    """
    df = historical_results.copy()
    
    # Actual vs predicted
    actual_win_rate = (df[result_col] == 'WIN').mean()
    predicted_avg_prob = df[prob_col].mean()
    calibration_error = actual_win_rate - predicted_avg_prob
    
    # Calculate empirical Kelly
    avg_odds = df[odds_col].mean()
    empirical_kelly = kelly_fraction(actual_win_rate, avg_odds)
    
    # Recommended adjustment
    if calibration_error < -0.05:
        recommendation = "Reduce Kelly (overconfident predictions)"
        suggested_fraction = 0.15
    elif calibration_error < 0:
        recommendation = "Use Quarter Kelly"
        suggested_fraction = 0.25
    elif calibration_error < 0.05:
        recommendation = "Use Half Kelly"
        suggested_fraction = 0.50
    else:
        recommendation = "Use 3/4 Kelly (underconfident predictions)"
        suggested_fraction = 0.75
    
    return {
        'actual_win_rate': actual_win_rate,
        'predicted_avg_prob': predicted_avg_prob,
        'calibration_error': calibration_error,
        'avg_odds': avg_odds,
        'empirical_kelly': empirical_kelly,
        'recommendation': recommendation,
        'suggested_fraction': suggested_fraction
    }


if __name__ == "__main__":
    print("Risk Analysis Module")
    print("="*50)
    
    # Demo: Kelly fraction calculation
    prob = 0.65
    odds = 1.80
    
    print(f"\nExample: P(win)={prob}, Odds={odds}")
    print(f"  Full Kelly: {kelly_fraction(prob, odds):.1%}")
    print(f"  Half Kelly: {fractional_kelly(prob, odds, 0.5):.1%}")
    print(f"  Quarter Kelly: {fractional_kelly(prob, odds, 0.25):.1%}")
