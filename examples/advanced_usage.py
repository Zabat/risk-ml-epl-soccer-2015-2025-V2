#!/usr/bin/env python3
"""
Advanced Usage Example
======================

This script demonstrates advanced features of the EPL Betting Model including:
- Custom configurations
- Risk analysis with CVaR
- Kelly criterion comparison
- Monte Carlo simulations
- Dynamic leverage strategies

Usage:
    python examples/advanced_usage.py

Author: Your Name
License: MIT
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import (
    EPLHomeWinPredictor,
    BettingBacktest,
    run_backtest,
    kelly_fraction,
    fractional_kelly,
    calculate_cvar,
    run_monte_carlo,
    analyze_kelly_variants,
    optimal_leverage_for_capital
)


def example_1_custom_configuration():
    """
    Example 1: Running backtest with custom configuration.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Custom Configuration")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/EPL_League_2015_2025.csv')
    
    # Conservative configuration
    conservative_config = {
        'min_prob': 0.75,        # Higher confidence threshold
        'min_value': 0.08,       # Higher edge required
        'min_odds': 1.20,
        'max_odds': 1.45,
        'min_elo_diff': 300,     # Stronger favorites only
        'kelly_fraction': 0.15,  # More conservative sizing
        'max_bet': 0.04,
    }
    
    # Aggressive configuration
    aggressive_config = {
        'min_prob': 0.60,
        'min_value': 0.03,
        'min_odds': 1.10,
        'max_odds': 1.70,
        'min_elo_diff': 150,
        'kelly_fraction': 0.50,
        'max_bet': 0.10,
    }
    
    print("\nConservative strategy:")
    backtest_cons, _ = run_backtest(df, conservative_config)
    stats_cons = backtest_cons.get_stats()
    if stats_cons:
        print(f"  Bets: {stats_cons['total_bets']}, ROI: {stats_cons['roi']:+.1%}")
    
    print("\nAggressive strategy:")
    backtest_agg, _ = run_backtest(df, aggressive_config)
    stats_agg = backtest_agg.get_stats()
    if stats_agg:
        print(f"  Bets: {stats_agg['total_bets']}, ROI: {stats_agg['roi']:+.1%}")


def example_2_kelly_analysis():
    """
    Example 2: Comparing different Kelly criterion variants.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Kelly Criterion Analysis")
    print("="*60)
    
    # Example betting opportunity
    prob = 0.70
    odds = 1.50
    
    print(f"\nScenario: P(win)={prob:.0%}, Odds={odds}")
    print("-" * 40)
    
    # Calculate different Kelly fractions
    full = kelly_fraction(prob, odds)
    half = fractional_kelly(prob, odds, 0.50)
    quarter = fractional_kelly(prob, odds, 0.25)
    eighth = fractional_kelly(prob, odds, 0.125)
    
    print(f"Full Kelly (100%):    {full:.1%} of bankroll")
    print(f"Half Kelly (50%):     {half:.1%} of bankroll")
    print(f"Quarter Kelly (25%):  {quarter:.1%} of bankroll")
    print(f"Eighth Kelly (12.5%): {eighth:.1%} of bankroll")
    
    # Expected growth rates (simplified)
    print("\nExpected log-growth per bet:")
    for name, frac in [('Full', full), ('Half', half), ('Quarter', quarter)]:
        if frac > 0:
            growth = prob * np.log(1 + frac * (odds - 1)) + (1 - prob) * np.log(1 - frac)
            print(f"  {name}: {growth:.4f}")


def example_3_monte_carlo():
    """
    Example 3: Monte Carlo simulation for risk analysis.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Monte Carlo Simulation")
    print("="*60)
    
    # Create sample betting opportunities
    np.random.seed(42)
    bet_opportunities = [
        {'prob': 0.70 + np.random.normal(0, 0.05), 
         'odds': 1.40 + np.random.normal(0, 0.1)}
        for _ in range(50)
    ]
    
    # Ensure valid ranges
    for bet in bet_opportunities:
        bet['prob'] = np.clip(bet['prob'], 0.55, 0.85)
        bet['odds'] = np.clip(bet['odds'], 1.20, 1.60)
    
    print(f"\nSimulating {len(bet_opportunities)} bets with 1000 iterations...")
    
    # Run simulation with different Kelly fractions
    results = {}
    for kelly_frac in [0.25, 0.50, 1.0]:
        sim = run_monte_carlo(
            bet_opportunities,
            n_simulations=1000,
            kelly_fraction=kelly_frac,
            random_seed=42
        )
        results[kelly_frac] = sim
        
        print(f"\n{kelly_frac:.0%} Kelly:")
        print(f"  Mean final: {sim['mean']:.0f}€")
        print(f"  Median:     {sim['median']:.0f}€")
        print(f"  VaR 95%:    {sim['var_95']:.0f}€")
        print(f"  CVaR 95%:   {sim['cvar_95']:.0f}€")
        print(f"  P(Profit):  {sim['prob_profit']:.1%}")


def example_4_dynamic_leverage():
    """
    Example 4: Dynamic leverage based on capital level.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Dynamic Leverage")
    print("="*60)
    
    initial_capital = 1000
    
    print("\nOptimal leverage by capital level:")
    print("-" * 50)
    
    capital_levels = [400, 600, 800, 1000, 1200, 1500, 2000, 3000]
    
    for capital in capital_levels:
        leverage = optimal_leverage_for_capital(
            capital, 
            initial_capital,
            max_leverage=3.0,
            risk_tolerance=0.7
        )
        pct = capital / initial_capital * 100
        print(f"  {capital:>5}€ ({pct:>5.0f}%): {leverage:.2f}x leverage")


def example_5_model_evaluation():
    """
    Example 5: Detailed model evaluation.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Model Evaluation")
    print("="*60)
    
    # Load data and split
    df = pd.read_csv('data/EPL_League_2015_2025.csv')
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    df = df.sort_values('MatchDate')
    
    # 80/20 temporal split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"\nTrain: {len(train_df)} matches")
    print(f"Test:  {len(test_df)} matches")
    
    # Train and evaluate
    predictor = EPLHomeWinPredictor()
    predictor.fit(train_df, verbose=False)
    
    eval_results = predictor.evaluate(test_df)
    
    print("\nTest Set Performance:")
    print("-" * 40)
    print(f"  Accuracy:   {eval_results['accuracy']:.1%}")
    print(f"  ROC-AUC:    {eval_results['roc_auc']:.3f}")
    print(f"  Brier:      {eval_results['brier_score']:.4f}")
    print(f"  Log Loss:   {eval_results['log_loss']:.4f}")
    
    cm = eval_results['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {cm['true_positive']}")
    print(f"  True Negatives:  {cm['true_negative']}")
    print(f"  False Positives: {cm['false_positive']}")
    print(f"  False Negatives: {cm['false_negative']}")


def example_6_feature_importance():
    """
    Example 6: Analyzing feature importance.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Feature Importance")
    print("="*60)
    
    df = pd.read_csv('data/EPL_League_2015_2025.csv')
    
    predictor = EPLHomeWinPredictor()
    predictor.fit(df, verbose=False)
    
    importance = predictor.get_feature_importance()
    
    print("\nFeature Coefficients (sorted by importance):")
    print("-" * 50)
    
    for _, row in importance.iterrows():
        bar = "█" * int(abs(row['coefficient']) * 10)
        sign = "+" if row['coefficient'] > 0 else "-"
        print(f"  {row['feature']:12s}: {sign}{abs(row['coefficient']):.3f} {bar}")


def main():
    """Run all examples."""
    print("\n" + "🎰"*30)
    print("    EPL BETTING MODEL - ADVANCED USAGE")
    print("🎰"*30)
    
    try:
        example_1_custom_configuration()
    except FileNotFoundError:
        print("\n⚠️ Data file not found. Skipping Example 1.")
    
    example_2_kelly_analysis()
    example_3_monte_carlo()
    example_4_dynamic_leverage()
    
    try:
        example_5_model_evaluation()
        example_6_feature_importance()
    except FileNotFoundError:
        print("\n⚠️ Data file not found. Skipping Examples 5 & 6.")
    
    print("\n" + "="*60)
    print("✅ ADVANCED USAGE EXAMPLES COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
