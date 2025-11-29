"""
Unit Tests for EPL Home Win Predictor
=====================================

Run tests:
    pytest tests/test_predictor.py -v

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

from src.predictor import EPLHomeWinPredictor


class TestEPLHomeWinPredictor:
    """Test cases for EPLHomeWinPredictor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 500
        
        home_elo = np.random.normal(1750, 100, n)
        away_elo = np.random.normal(1750, 100, n)
        elo_diff = home_elo - away_elo
        
        # Generate results based on Elo difference
        home_win_prob = 1 / (1 + np.exp(-elo_diff / 200))
        results = np.where(np.random.random(n) < home_win_prob, 'H',
                          np.where(np.random.random(n) < 0.3, 'D', 'A'))
        
        return pd.DataFrame({
            'HomeElo': home_elo,
            'AwayElo': away_elo,
            'Form3Home': np.random.uniform(0, 9, n),
            'Form5Home': np.random.uniform(0, 15, n),
            'Form3Away': np.random.uniform(0, 9, n),
            'Form5Away': np.random.uniform(0, 15, n),
            'FTResult': results,
            'HomeTeam': ['TeamA'] * n,
            'AwayTeam': ['TeamB'] * n,
        })
    
    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return EPLHomeWinPredictor()
    
    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor is not None
        assert predictor.is_fitted == False
        assert len(predictor.feature_names) == 7
    
    def test_fit(self, predictor, sample_data):
        """Test model fitting."""
        predictor.fit(sample_data, verbose=False)
        
        assert predictor.is_fitted == True
        assert 'accuracy' in predictor.training_stats
        assert predictor.training_stats['n_samples'] > 0
    
    def test_predict_proba(self, predictor, sample_data):
        """Test probability prediction."""
        predictor.fit(sample_data, verbose=False)
        
        probs = predictor.predict_proba(sample_data.head(10))
        
        assert len(probs) == 10
        assert all(0 <= p <= 1 for p in probs)
    
    def test_predict(self, predictor, sample_data):
        """Test binary prediction."""
        predictor.fit(sample_data, verbose=False)
        
        predictions = predictor.predict(sample_data.head(10))
        
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_match(self, predictor, sample_data):
        """Test single match prediction."""
        predictor.fit(sample_data, verbose=False)
        
        result = predictor.predict_match(
            home_elo=1900,
            away_elo=1700
        )
        
        assert 'prob_home_win' in result
        assert 'confidence' in result
        assert 'recommendation' in result
        assert 0 <= result['prob_home_win'] <= 1
    
    def test_predict_unfitted_raises(self, predictor, sample_data):
        """Test that prediction on unfitted model raises error."""
        with pytest.raises(ValueError):
            predictor.predict_proba(sample_data)
    
    def test_feature_importance(self, predictor, sample_data):
        """Test feature importance retrieval."""
        predictor.fit(sample_data, verbose=False)
        
        importance = predictor.get_feature_importance()
        
        assert len(importance) == len(predictor.feature_names)
        assert 'feature' in importance.columns
        assert 'coefficient' in importance.columns
    
    def test_save_load(self, predictor, sample_data, tmp_path):
        """Test model save and load."""
        predictor.fit(sample_data, verbose=False)
        
        # Save
        save_path = tmp_path / "test_model.pkl"
        predictor.save(str(save_path))
        
        assert save_path.exists()
        
        # Load
        loaded = EPLHomeWinPredictor.load(str(save_path))
        
        assert loaded.is_fitted == True
        assert len(loaded.feature_names) == len(predictor.feature_names)
        
        # Compare predictions
        orig_probs = predictor.predict_proba(sample_data.head(5))
        loaded_probs = loaded.predict_proba(sample_data.head(5))
        
        np.testing.assert_array_almost_equal(orig_probs, loaded_probs)
    
    def test_evaluate(self, predictor, sample_data):
        """Test model evaluation."""
        predictor.fit(sample_data.iloc[:400], verbose=False)
        
        eval_results = predictor.evaluate(sample_data.iloc[400:])
        
        assert 'accuracy' in eval_results
        assert 'roc_auc' in eval_results
        assert 'confusion_matrix' in eval_results
        assert 0 <= eval_results['accuracy'] <= 1
    
    def test_high_elo_diff_prediction(self, predictor, sample_data):
        """Test that high Elo diff produces high probability."""
        predictor.fit(sample_data, verbose=False)
        
        # Large home advantage
        result_high = predictor.predict_match(home_elo=2000, away_elo=1500)
        
        # Large away advantage
        result_low = predictor.predict_match(home_elo=1500, away_elo=2000)
        
        assert result_high['prob_home_win'] > result_low['prob_home_win']
        assert result_high['prob_home_win'] > 0.7
        assert result_low['prob_home_win'] < 0.3


class TestPrepareFeatures:
    """Test feature preparation."""
    
    def test_elo_diff_calculation(self):
        """Test EloDiff feature calculation."""
        predictor = EPLHomeWinPredictor()
        
        df = pd.DataFrame({
            'HomeElo': [1800],
            'AwayElo': [1700],
            'Form3Home': [5],
            'Form5Home': [8],
            'Form3Away': [4],
            'Form5Away': [7]
        })
        
        prepared = predictor._prepare_features(df)
        
        assert 'EloDiff' in prepared.columns
        assert prepared['EloDiff'].iloc[0] == 100
    
    def test_form_diff_calculation(self):
        """Test FormDiff feature calculation."""
        predictor = EPLHomeWinPredictor()
        
        df = pd.DataFrame({
            'HomeElo': [1800],
            'AwayElo': [1700],
            'Form3Home': [5],
            'Form5Home': [10],
            'Form3Away': [4],
            'Form5Away': [6]
        })
        
        prepared = predictor._prepare_features(df)
        
        assert 'FormDiff' in prepared.columns
        assert prepared['FormDiff'].iloc[0] == 4  # 10 - 6


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
