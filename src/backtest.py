"""
Betting Backtest Engine
=======================

Comprehensive backtesting framework for sports betting strategies with
Kelly Criterion money management and performance analytics.

Example:
    >>> from src.backtest import BettingBacktest, run_backtest
    >>> backtest, predictor = run_backtest(df, config)
    >>> print(backtest.get_stats())

Author: Your Name
License: MIT
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .predictor import EPLHomeWinPredictor

warnings.filterwarnings('ignore')


@dataclass
class BetRecord:
    """Record of a single bet."""
    date: datetime
    home_team: str
    away_team: str
    bet_type: str  # 'H', 'D', or 'A'
    odds: float
    prob_predicted: float
    value: float  # Expected value = prob * odds - 1
    elo_diff: float
    bet_fraction: float
    bet_amount: float
    result: str  # 'WIN' or 'LOSS'
    profit: float
    bankroll_after: float


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Betting criteria
    min_prob: float = 0.68
    min_value: float = 0.05
    min_odds: float = 1.15
    max_odds: float = 1.55
    min_elo_diff: float = 250
    
    # Money management
    kelly_fraction: float = 0.25
    max_bet: float = 0.06
    min_bet: float = 0.005
    initial_bankroll: float = 1000
    stop_loss: float = 0.10  # Stop if bankroll drops below this fraction
    
    # Model settings
    train_window: int = 300
    retrain_every: int = 50
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'BacktestConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class BettingBacktest:
    """
    Backtesting engine for sports betting strategies.
    
    This class handles:
    - Bankroll tracking and management
    - Bet execution and recording
    - Performance metrics calculation
    - Risk analysis
    
    Attributes:
        initial_bankroll: Starting capital
        bankroll: Current capital
        bankroll_history: List of bankroll values over time
        bet_history: List of BetRecord objects
        dates: List of dates for each bankroll snapshot
    
    Example:
        >>> backtest = BettingBacktest(initial_bankroll=1000)
        >>> backtest.place_bet(match_info, 'H', 1.50, 0.75, 0.05)
        >>> print(backtest.get_stats())
    """
    
    def __init__(self, initial_bankroll: float = 1000):
        """
        Initialize the backtest engine.
        
        Args:
            initial_bankroll: Starting capital in base currency
        """
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.bankroll_history: List[float] = [initial_bankroll]
        self.bet_history: List[BetRecord] = []
        self.dates: List[datetime] = []
        
    def calculate_kelly(
        self,
        prob: float,
        odds: float,
        fraction: float = 0.25
    ) -> float:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        The Kelly Criterion maximizes the expected logarithm of wealth:
        
        f* = (p * b - q) / b
        
        where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = net odds received (decimal odds - 1)
        
        Args:
            prob: Estimated probability of winning (0 to 1)
            odds: Decimal odds (e.g., 1.50 for 1.50:1)
            fraction: Kelly fraction to use (0.25 = quarter Kelly)
            
        Returns:
            Optimal bet fraction of bankroll (0 to 1)
            
        Note:
            Quarter Kelly (0.25) is recommended for most applications
            as it significantly reduces variance while sacrificing
            only ~25% of expected growth rate.
        """
        if odds <= 1 or prob <= 0 or prob >= 1:
            return 0.0
        
        q = 1 - prob
        b = odds - 1
        
        # Full Kelly formula
        kelly = (prob * b - q) / b
        
        # Ensure non-negative
        kelly = max(0.0, kelly)
        
        # Apply fractional Kelly
        return kelly * fraction
    
    def place_bet(
        self,
        match_info: pd.Series,
        bet_type: str,
        odds: float,
        prob: float,
        bet_fraction: float
    ) -> bool:
        """
        Execute a bet and update bankroll.
        
        Args:
            match_info: Series with match details (HomeTeam, AwayTeam, FTResult, etc.)
            bet_type: Type of bet ('H' for home, 'D' for draw, 'A' for away)
            odds: Decimal odds for this bet
            prob: Our estimated probability
            bet_fraction: Fraction of bankroll to bet
            
        Returns:
            bool: True if bet won, False otherwise
        """
        if bet_fraction <= 0:
            return False
        
        bet_amount = self.bankroll * bet_fraction
        actual_result = match_info['FTResult']
        is_win = (bet_type == actual_result)
        
        # Calculate profit/loss
        if is_win:
            profit = bet_amount * (odds - 1)
            self.bankroll += profit
        else:
            profit = -bet_amount
            self.bankroll -= bet_amount
        
        # Record the bet
        bet_record = BetRecord(
            date=match_info['MatchDate'],
            home_team=match_info['HomeTeam'],
            away_team=match_info['AwayTeam'],
            bet_type=bet_type,
            odds=odds,
            prob_predicted=prob,
            value=prob * odds - 1,
            elo_diff=match_info.get('HomeElo', 0) - match_info.get('AwayElo', 0),
            bet_fraction=bet_fraction,
            bet_amount=bet_amount,
            result='WIN' if is_win else 'LOSS',
            profit=profit,
            bankroll_after=self.bankroll
        )
        self.bet_history.append(bet_record)
        
        return is_win
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """
        Calculate comprehensive performance statistics.
        
        Returns:
            Dictionary containing:
                - total_bets: Number of bets placed
                - wins/losses: Win and loss counts
                - win_rate: Percentage of winning bets
                - avg_odds: Average odds of bets
                - avg_value: Average expected value
                - total_profit: Net profit in currency
                - total_staked: Total amount wagered
                - roi: Return on investment
                - final_bankroll: Ending capital
                - return_multiple: Final / Initial bankroll
                - max_drawdown: Largest peak-to-trough decline
                - cagr: Compound annual growth rate
                - sharpe_ratio: Risk-adjusted return
                - avg_bet_size: Average bet as fraction of bankroll
                - max_bet_size: Largest bet as fraction of bankroll
            
            Returns None if no bets have been placed.
        """
        if not self.bet_history:
            return None
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([vars(b) for b in self.bet_history])
        
        wins = (df['result'] == 'WIN').sum()
        total = len(df)
        
        total_profit = df['profit'].sum()
        total_staked = df['bet_amount'].sum()
        
        # Calculate maximum drawdown
        peak = self.initial_bankroll
        max_drawdown = 0.0
        drawdowns = []
        
        for br in self.bankroll_history:
            if br > peak:
                peak = br
            drawdown = (peak - br) / peak if peak > 0 else 0
            drawdowns.append(drawdown)
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate CAGR
        if len(df) > 0:
            days = (df['date'].max() - df['date'].min()).days
            years = max(0.5, days / 365.25)
            cagr = (self.bankroll / self.initial_bankroll) ** (1/years) - 1
        else:
            cagr = 0.0
        
        # Calculate Sharpe Ratio (annualized)
        returns = df['profit'] / df['bet_amount']
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        return {
            'total_bets': total,
            'wins': int(wins),
            'losses': int(total - wins),
            'win_rate': float(wins / total) if total > 0 else 0,
            'avg_odds': float(df['odds'].mean()),
            'avg_value': float(df['value'].mean()),
            'total_profit': float(total_profit),
            'total_staked': float(total_staked),
            'roi': float(total_profit / total_staked) if total_staked > 0 else 0,
            'final_bankroll': float(self.bankroll),
            'return_multiple': float(self.bankroll / self.initial_bankroll),
            'max_drawdown': float(max_drawdown),
            'cagr': float(cagr),
            'sharpe_ratio': float(sharpe),
            'avg_bet_size': float(df['bet_fraction'].mean()),
            'max_bet_size': float(df['bet_fraction'].max())
        }
    
    def get_bet_history_df(self) -> pd.DataFrame:
        """Get bet history as a DataFrame."""
        if not self.bet_history:
            return pd.DataFrame()
        return pd.DataFrame([vars(b) for b in self.bet_history])
    
    def reset(self) -> None:
        """Reset the backtest to initial state."""
        self.bankroll = self.initial_bankroll
        self.bankroll_history = [self.initial_bankroll]
        self.bet_history = []
        self.dates = []


def run_backtest(
    df: pd.DataFrame,
    config: Optional[Dict] = None
) -> Tuple[BettingBacktest, EPLHomeWinPredictor]:
    """
    Execute a complete backtest with given configuration.
    
    This function:
    1. Prepares and validates the data
    2. Trains the prediction model with walk-forward validation
    3. Places bets according to the strategy
    4. Tracks performance metrics
    
    Args:
        df: DataFrame with historical match data
        config: Configuration dictionary (uses defaults if None)
        
    Returns:
        Tuple of (BettingBacktest, EPLHomeWinPredictor)
        
    Example:
        >>> config = {'min_prob': 0.70, 'kelly_fraction': 0.25}
        >>> backtest, predictor = run_backtest(df, config)
        >>> print(backtest.get_stats())
    """
    # Parse configuration
    if config is None:
        cfg = BacktestConfig()
    elif isinstance(config, BacktestConfig):
        cfg = config
    else:
        cfg = BacktestConfig.from_dict(config)
    
    print("="*70)
    print("🎰 EPL BETTING BACKTEST")
    print("="*70)
    
    # Data preparation
    df = df.copy()
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    df = df.sort_values('MatchDate').reset_index(drop=True)
    
    # Validate required columns
    required = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home',
                'Form3Away', 'Form5Away', 'FTResult', 'OddHome']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df_clean = df.dropna(subset=required).reset_index(drop=True)
    
    print(f"\n📊 Data: {len(df_clean)} matches")
    print(f"   Period: {df_clean['MatchDate'].min().date()} → {df_clean['MatchDate'].max().date()}")
    
    # Initialize components
    predictor = EPLHomeWinPredictor()
    backtest = BettingBacktest(initial_bankroll=cfg.initial_bankroll)
    
    last_train = 0
    
    # Print configuration
    print(f"\n⚙️  Configuration:")
    print(f"   Min Prob: {cfg.min_prob:.0%} | Min Value: {cfg.min_value:.0%}")
    print(f"   Odds Range: {cfg.min_odds:.2f} - {cfg.max_odds:.2f}")
    print(f"   Min Elo Diff: {cfg.min_elo_diff:.0f}")
    print(f"   Kelly Fraction: {cfg.kelly_fraction:.0%} | Max Bet: {cfg.max_bet:.0%}")
    
    print(f"\n🔄 Running backtest...")
    
    # Main backtest loop
    for i in range(cfg.train_window, len(df_clean)):
        # Retrain model periodically
        if i - last_train >= cfg.retrain_every or not predictor.is_fitted:
            train_data = df_clean.iloc[:i]
            predictor.fit(train_data, verbose=False)
            last_train = i
        
        # Get current match
        current_match = df_clean.iloc[[i]]
        match_info = df_clean.iloc[i]
        
        # Generate prediction
        prob_home = predictor.predict_proba(current_match)[0]
        odds_home = match_info['OddHome']
        value = prob_home * odds_home - 1
        elo_diff = match_info['HomeElo'] - match_info['AwayElo']
        
        # Check betting criteria
        should_bet = (
            prob_home >= cfg.min_prob and
            value >= cfg.min_value and
            cfg.min_odds <= odds_home <= cfg.max_odds and
            elo_diff >= cfg.min_elo_diff
        )
        
        if should_bet:
            # Calculate bet size
            kelly = backtest.calculate_kelly(prob_home, odds_home, cfg.kelly_fraction)
            bet_fraction = min(kelly, cfg.max_bet)
            
            if bet_fraction >= cfg.min_bet:
                backtest.place_bet(match_info, 'H', odds_home, prob_home, bet_fraction)
        
        # Record bankroll state
        backtest.bankroll_history.append(backtest.bankroll)
        backtest.dates.append(match_info['MatchDate'])
        
        # Check stop-loss
        if backtest.bankroll < cfg.initial_bankroll * cfg.stop_loss:
            print(f"   ⚠️ Stop-loss triggered at match {i}")
            break
    
    # Print summary
    stats = backtest.get_stats()
    if stats:
        print(f"\n✅ Backtest complete: {stats['total_bets']} bets")
        print(f"   Win Rate: {stats['win_rate']:.1%}")
        print(f"   Final Bankroll: {stats['final_bankroll']:.0f}€")
        print(f"   ROI: {stats['roi']:+.1%}")
    else:
        print("\n❌ No bets placed with current configuration")
    
    return backtest, predictor


def print_performance_report(backtest: BettingBacktest) -> None:
    """
    Print a comprehensive performance report.
    
    Args:
        backtest: BettingBacktest instance with completed backtest
    """
    stats = backtest.get_stats()
    if not stats:
        print("❌ No statistics available")
        return
    
    print(f"""
{'='*70}
📊 PERFORMANCE REPORT
{'='*70}

┌─────────────────────────────────────────────────────────────────────┐
│                        BACKTEST RESULTS                             │
├─────────────────────────────────────────────────────────────────────┤
│  💰 BANKROLL                                                        │
│     Initial Capital:     {backtest.initial_bankroll:>10.0f} €                            │
│     Final Capital:       {stats['final_bankroll']:>10.0f} €                            │
│     Return Multiple:     {stats['return_multiple']:>10.2f} x                            │
│     Total Profit:        {stats['total_profit']:>+10.0f} €                            │
├─────────────────────────────────────────────────────────────────────┤
│  📈 PERFORMANCE METRICS                                             │
│     ROI:                 {stats['roi']*100:>+10.1f} %                            │
│     CAGR:                {stats['cagr']*100:>+10.1f} %                            │
│     Sharpe Ratio:        {stats['sharpe_ratio']:>10.2f}                              │
│     Max Drawdown:        {stats['max_drawdown']*100:>10.1f} %                            │
├─────────────────────────────────────────────────────────────────────┤
│  🎯 BETTING STATISTICS                                              │
│     Total Bets:          {stats['total_bets']:>10d}                              │
│     Wins:                {stats['wins']:>10d} ({stats['win_rate']*100:.1f}%)                        │
│     Losses:              {stats['losses']:>10d}                              │
│     Average Odds:        {stats['avg_odds']:>10.2f}                              │
│     Average Value:       {stats['avg_value']*100:>+10.1f} %                            │
│     Avg Bet Size:        {stats['avg_bet_size']*100:>10.1f} %                            │
│     Max Bet Size:        {stats['max_bet_size']*100:>10.1f} %                            │
└─────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    print("Betting Backtest Engine")
    print("="*50)
    print("Import and use run_backtest() to execute a backtest.")
