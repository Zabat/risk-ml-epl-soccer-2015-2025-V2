#!/usr/bin/env python3
"""
Quick Start Example
===================

This script demonstrates the basic usage of the EPL Betting Model.

Usage:
    python examples/quick_start.py

Author: Your Name
License: MIT
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src import EPLHomeWinPredictor, run_backtest


def main():
    """Main function demonstrating basic usage."""
    
    print("="*60)
    print("EPL HOME WIN PREDICTOR - QUICK START")
    print("="*60)
    
    # ================================================
    # 1. LOAD DATA
    # ================================================
    print("\n1. Loading data...")
    
    try:
        df = pd.read_csv('data/EPL_League_2015_2025.csv')
        print(f"   ✓ Loaded {len(df)} matches")
    except FileNotFoundError:
        print("   ❌ Data file not found!")
        print("   Please place EPL_League_2015_2025.csv in the data/ folder")
        return
    
    # ================================================
    # 2. TRAIN MODEL
    # ================================================
    print("\n2. Training model...")
    
    predictor = EPLHomeWinPredictor()
    predictor.fit(df)
    
    # ================================================
    # 3. MAKE PREDICTIONS
    # ================================================
    print("\n3. Making predictions...")
    
    # Example matches
    matches = [
        ("Man City vs Southampton", 1950, 1550),
        ("Liverpool vs Arsenal", 2000, 1980),
        ("Wolves vs Brighton", 1740, 1780),
    ]
    
    print("\n   Match Predictions:")
    print("   " + "-"*50)
    
    for name, home_elo, away_elo in matches:
        result = predictor.predict_match(home_elo, away_elo)
        print(f"   {name}")
        print(f"      P(Home Win): {result['prob_home_win']:.1%}")
        print(f"      {result['recommendation']}")
        print()
    
    # ================================================
    # 4. RUN BACKTEST
    # ================================================
    print("4. Running backtest...")
    
    config = {
        'min_prob': 0.68,
        'min_value': 0.05,
        'min_odds': 1.15,
        'max_odds': 1.55,
        'min_elo_diff': 250,
        'kelly_fraction': 0.25,
        'max_bet': 0.06,
    }
    
    backtest, _ = run_backtest(df, config)
    
    # ================================================
    # 5. DISPLAY RESULTS
    # ================================================
    stats = backtest.get_stats()
    
    if stats:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"""
   Total Bets:      {stats['total_bets']}
   Win Rate:        {stats['win_rate']:.1%}
   Final Bankroll:  {stats['final_bankroll']:.0f}€
   ROI:             {stats['roi']:+.1%}
   Max Drawdown:    {stats['max_drawdown']:.1%}
        """)
    
    # ================================================
    # 6. SAVE MODEL
    # ================================================
    print("6. Saving model...")
    
    predictor.save('models/epl_home_win_model.pkl')
    
    print("\n✅ Quick start complete!")
    print("\nNext steps:")
    print("   - Explore notebooks/exploration.ipynb")
    print("   - Run advanced_usage.py for more features")
    print("   - Check docs/ for detailed documentation")


if __name__ == "__main__":
    main()
