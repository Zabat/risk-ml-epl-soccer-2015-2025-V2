"""
Unit Tests for Betting Backtest Engine
======================================

Run tests:
    pytest tests/test_backtest.py -v

Author: Your Name
License: MIT
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest import BettingBacktest, BacktestConfig


class TestBettingBacktest:
    """Test cases for BettingBacktest class."""
    
    @pytest.fixture
    def backtest(self):
        """Create backtest instance."""
        return BettingBacktest(initial_bankroll=1000)
    
    def test_initialization(self, backtest):
        """Test backtest initialization."""
        assert backtest.initial_bankroll == 1000
        assert backtest.bankroll == 1000
        assert len(backtest.bankroll_history) == 1
        assert len(backtest.bet_history) == 0
    
    def test_calculate_kelly_positive_edge(self, backtest):
        """Test Kelly calculation with positive edge."""
        # prob * odds > 1 means positive edge
        kelly = backtest.calculate_kelly(prob=0.70, odds=1.50, fraction=1.0)
        
        assert kelly > 0
        assert kelly < 1
    
    def test_calculate_kelly_no_edge(self, backtest):
        """Test Kelly calculation with no edge."""
        # prob * odds = 1 means no edge
        kelly = backtest.calculate_kelly(prob=0.50, odds=2.00, fraction=1.0)
        
        assert kelly == 0
    
    def test_calculate_kelly_negative_edge(self, backtest):
        """Test Kelly calculation with negative edge."""
        # prob * odds < 1 means negative edge
        kelly = backtest.calculate_kelly(prob=0.40, odds=2.00, fraction=1.0)
        
        assert kelly == 0  # Should not bet
    
    def test_calculate_kelly_fractional(self, backtest):
        """Test fractional Kelly."""
        full_kelly = backtest.calculate_kelly(prob=0.70, odds=1.50, fraction=1.0)
        half_kelly = backtest.calculate_kelly(prob=0.70, odds=1.50, fraction=0.5)
        quarter_kelly = backtest.calculate_kelly(prob=0.70, odds=1.50, fraction=0.25)
        
        assert half_kelly == full_kelly * 0.5
        assert quarter_kelly == full_kelly * 0.25
    
    def test_place_winning_bet(self, backtest):
        """Test placing a winning bet."""
        match_info = pd.Series({
            'MatchDate': pd.Timestamp('2024-01-01'),
            'HomeTeam': 'TeamA',
            'AwayTeam': 'TeamB',
            'FTResult': 'H',  # Home win
            'HomeElo': 1800,
            'AwayElo': 1700
        })
        
        initial_bankroll = backtest.bankroll
        result = backtest.place_bet(match_info, 'H', odds=1.50, prob=0.70, bet_fraction=0.10)
        
        assert result == True  # Bet won
        assert backtest.bankroll > initial_bankroll
        assert len(backtest.bet_history) == 1
        assert backtest.bet_history[0].result == 'WIN'
    
    def test_place_losing_bet(self, backtest):
        """Test placing a losing bet."""
        match_info = pd.Series({
            'MatchDate': pd.Timestamp('2024-01-01'),
            'HomeTeam': 'TeamA',
            'AwayTeam': 'TeamB',
            'FTResult': 'A',  # Away win
            'HomeElo': 1800,
            'AwayElo': 1700
        })
        
        initial_bankroll = backtest.bankroll
        result = backtest.place_bet(match_info, 'H', odds=1.50, prob=0.70, bet_fraction=0.10)
        
        assert result == False  # Bet lost
        assert backtest.bankroll < initial_bankroll
        assert len(backtest.bet_history) == 1
        assert backtest.bet_history[0].result == 'LOSS'
    
    def test_get_stats_empty(self, backtest):
        """Test stats with no bets."""
        stats = backtest.get_stats()
        assert stats is None
    
    def test_get_stats_with_bets(self, backtest):
        """Test stats after placing bets."""
        # Place some bets
        for result in ['H', 'H', 'A', 'H']:  # 3 wins, 1 loss
            match_info = pd.Series({
                'MatchDate': pd.Timestamp('2024-01-01'),
                'HomeTeam': 'TeamA',
                'AwayTeam': 'TeamB',
                'FTResult': result,
                'HomeElo': 1800,
                'AwayElo': 1700
            })
            backtest.place_bet(match_info, 'H', odds=1.50, prob=0.70, bet_fraction=0.05)
        
        stats = backtest.get_stats()
        
        assert stats is not None
        assert stats['total_bets'] == 4
        assert stats['wins'] == 3
        assert stats['losses'] == 1
        assert stats['win_rate'] == 0.75
        assert 'roi' in stats
        assert 'max_drawdown' in stats
    
    def test_reset(self, backtest):
        """Test reset functionality."""
        # Place a bet
        match_info = pd.Series({
            'MatchDate': pd.Timestamp('2024-01-01'),
            'HomeTeam': 'TeamA',
            'AwayTeam': 'TeamB',
            'FTResult': 'H',
            'HomeElo': 1800,
            'AwayElo': 1700
        })
        backtest.place_bet(match_info, 'H', odds=1.50, prob=0.70, bet_fraction=0.10)
        
        assert len(backtest.bet_history) > 0
        
        # Reset
        backtest.reset()
        
        assert backtest.bankroll == backtest.initial_bankroll
        assert len(backtest.bet_history) == 0
        assert len(backtest.bankroll_history) == 1


class TestBacktestConfig:
    """Test cases for BacktestConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = BacktestConfig()
        
        assert config.min_prob == 0.68
        assert config.kelly_fraction == 0.25
        assert config.initial_bankroll == 1000
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            min_prob=0.75,
            kelly_fraction=0.50,
            initial_bankroll=5000
        )
        
        assert config.min_prob == 0.75
        assert config.kelly_fraction == 0.50
        assert config.initial_bankroll == 5000
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'min_prob': 0.80,
            'max_bet': 0.10,
            'unknown_key': 'ignored'
        }
        
        config = BacktestConfig.from_dict(config_dict)
        
        assert config.min_prob == 0.80
        assert config.max_bet == 0.10
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = BacktestConfig(min_prob=0.75)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'min_prob' in config_dict
        assert config_dict['min_prob'] == 0.75


class TestKellyFormula:
    """Test Kelly Criterion edge cases."""
    
    @pytest.fixture
    def backtest(self):
        return BettingBacktest()
    
    def test_kelly_boundary_odds_1(self, backtest):
        """Test Kelly with odds = 1 (no profit possible)."""
        kelly = backtest.calculate_kelly(prob=0.70, odds=1.0, fraction=1.0)
        assert kelly == 0
    
    def test_kelly_boundary_prob_0(self, backtest):
        """Test Kelly with probability = 0."""
        kelly = backtest.calculate_kelly(prob=0, odds=2.0, fraction=1.0)
        assert kelly == 0
    
    def test_kelly_boundary_prob_1(self, backtest):
        """Test Kelly with probability = 1 (sure win)."""
        kelly = backtest.calculate_kelly(prob=1.0, odds=2.0, fraction=1.0)
        # Should still return a valid fraction
        assert kelly >= 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
