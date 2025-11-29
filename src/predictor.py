"""
EPL Home Win Predictor
======================

A Logistic Regression model for predicting English Premier League home wins
using Elo ratings and team form features.

Example:
    >>> from src.predictor import EPLHomeWinPredictor
    >>> predictor = EPLHomeWinPredictor()
    >>> predictor.fit(training_data)
    >>> result = predictor.predict_match(home_elo=1900, away_elo=1700)
    >>> print(f"P(Home Win): {result['prob_home_win']:.1%}")

Author: Your Name
License: MIT
"""

import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


class EPLHomeWinPredictor:
    """
    Logistic Regression model for predicting EPL home wins.
    
    This model uses Elo ratings and recent form to predict the probability
    of a home team winning. It's designed for high-confidence predictions
    on matches with significant Elo advantages.
    
    Attributes:
        model: Trained LogisticRegression model
        scaler: StandardScaler for feature normalization
        feature_names: List of feature column names
        is_fitted: Whether the model has been trained
        training_stats: Dictionary of training statistics
    
    Example:
        >>> predictor = EPLHomeWinPredictor()
        >>> predictor.fit(df)
        >>> prob = predictor.predict_proba(test_df)
    """
    
    # Class constants
    DEFAULT_FEATURES = [
        'EloDiff',      # Elo rating difference (Home - Away)
        'EloSum',       # Combined Elo ratings (match quality)
        'Form3Home',    # Home team points from last 3 matches
        'Form5Home',    # Home team points from last 5 matches
        'Form3Away',    # Away team points from last 3 matches
        'Form5Away',    # Away team points from last 5 matches
        'FormDiff',     # Form difference (Home5 - Away5)
    ]
    
    CONFIDENCE_THRESHOLDS = {
        'high': 0.70,
        'medium': 0.55,
        'low': 0.45,
    }
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        features: Optional[List[str]] = None
    ):
        """
        Initialize the EPL Home Win Predictor.
        
        Args:
            C: Inverse regularization strength. Smaller values = stronger regularization.
            max_iter: Maximum iterations for solver convergence.
            random_state: Random seed for reproducibility.
            features: Custom feature list. Defaults to DEFAULT_FEATURES.
        """
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs',
            class_weight=None  # Balanced classes not needed for this task
        )
        self.scaler = StandardScaler()
        self.feature_names = features or self.DEFAULT_FEATURES.copy()
        self.is_fitted = False
        self.training_stats: Dict = {}
        
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features from raw match data.
        
        This method computes:
        - EloDiff: Difference in Elo ratings (positive = home advantage)
        - EloSum: Total Elo (proxy for match quality/importance)
        - FormDiff: Difference in recent form
        
        Args:
            df: DataFrame with HomeElo, AwayElo, and Form columns
            
        Returns:
            DataFrame with additional computed features
        """
        df_prep = df.copy()
        
        # Compute derived features
        if 'EloDiff' not in df_prep.columns:
            df_prep['EloDiff'] = df_prep['HomeElo'] - df_prep['AwayElo']
            
        if 'EloSum' not in df_prep.columns:
            df_prep['EloSum'] = df_prep['HomeElo'] + df_prep['AwayElo']
            
        if 'FormDiff' not in df_prep.columns:
            df_prep['FormDiff'] = df_prep['Form5Home'] - df_prep['Form5Away']
        
        return df_prep
    
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'FTResult',
        verbose: bool = True
    ) -> 'EPLHomeWinPredictor':
        """
        Train the model on historical match data.
        
        Args:
            df: DataFrame with match data including features and results
            target_col: Column containing match result ('H', 'D', 'A')
            verbose: Whether to print training statistics
            
        Returns:
            self: Fitted predictor instance
            
        Raises:
            ValueError: If required columns are missing
        """
        # Prepare features
        df_prep = self._prepare_features(df)
        
        # Create binary target (1 = Home Win, 0 = Not Home Win)
        df_prep['HomeWin'] = (df_prep[target_col] == 'H').astype(int)
        
        # Validate and clean data
        required_cols = self.feature_names + ['HomeWin']
        missing_cols = set(required_cols) - set(df_prep.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df_clean = df_prep.dropna(subset=required_cols)
        
        if len(df_clean) < 100:
            raise ValueError(f"Insufficient data: {len(df_clean)} samples (need >= 100)")
        
        # Extract features and target
        X = df_clean[self.feature_names].values
        y = df_clean['HomeWin'].values
        
        # Fit scaler and transform features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Calculate and store training statistics
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        self.training_stats = {
            'n_samples': len(y),
            'n_positive': int(y.sum()),
            'n_negative': int(len(y) - y.sum()),
            'accuracy': float(accuracy_score(y, y_pred)),
            'roc_auc': float(roc_auc_score(y, y_prob)),
            'brier_score': float(brier_score_loss(y, y_prob)),
            'log_loss': float(log_loss(y, y_prob)),
            'home_win_rate': float(y.mean()),
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.coef_[0].tolist()
            ))
        }
        
        if verbose:
            self._print_training_summary()
        
        return self
    
    def _print_training_summary(self) -> None:
        """Print a summary of training results."""
        stats = self.training_stats
        print(f"\n{'='*60}")
        print(f"MODEL TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"  Samples: {stats['n_samples']:,} ({stats['n_positive']:,} wins, {stats['n_negative']:,} other)")
        print(f"  Home Win Rate: {stats['home_win_rate']:.1%}")
        print(f"\n  Performance Metrics:")
        print(f"    Accuracy:    {stats['accuracy']:.1%}")
        print(f"    ROC-AUC:     {stats['roc_auc']:.3f}")
        print(f"    Brier Score: {stats['brier_score']:.4f}")
        print(f"    Log Loss:    {stats['log_loss']:.4f}")
        print(f"\n  Feature Importance:")
        for feat, coef in sorted(stats['feature_importance'].items(), 
                                  key=lambda x: abs(x[1]), reverse=True):
            direction = "↑" if coef > 0 else "↓"
            print(f"    {feat:12s}: {coef:+.3f} {direction}")
        print(f"{'='*60}\n")
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict home win probability for given matches.
        
        Args:
            df: DataFrame with match features
            
        Returns:
            Array of probabilities (between 0 and 1)
            
        Raises:
            ValueError: If model hasn't been fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        df_prep = self._prepare_features(df)
        
        # Validate features
        missing = set(self.feature_names) - set(df_prep.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        X = df_prep[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict home win (binary) for given matches.
        
        Args:
            df: DataFrame with match features
            threshold: Probability threshold for positive prediction
            
        Returns:
            Array of predictions (1 = Home Win, 0 = Not Home Win)
        """
        probs = self.predict_proba(df)
        return (probs >= threshold).astype(int)
    
    def predict_match(
        self,
        home_elo: float,
        away_elo: float,
        form3_home: float = 4.5,
        form5_home: float = 7.5,
        form3_away: float = 4.5,
        form5_away: float = 7.5
    ) -> Dict:
        """
        Predict a single match with detailed analysis.
        
        This is a convenience method for predicting individual matches
        without needing to construct a DataFrame.
        
        Args:
            home_elo: Elo rating of home team (typically 1400-2100)
            away_elo: Elo rating of away team
            form3_home: Home team points from last 3 matches (0-9)
            form5_home: Home team points from last 5 matches (0-15)
            form3_away: Away team points from last 3 matches (0-9)
            form5_away: Away team points from last 5 matches (0-15)
            
        Returns:
            Dictionary containing:
                - prob_home_win: Probability of home win
                - prob_not_home_win: Probability of draw or away win
                - prediction: 'HOME WIN' or 'NO HOME WIN'
                - confidence: 'HIGH', 'MEDIUM', 'LOW', or 'INVERSE'
                - recommendation: Betting recommendation string
                - elo_diff: Elo difference (home - away)
                - form_diff: Form difference
                
        Example:
            >>> result = predictor.predict_match(1900, 1700)
            >>> print(result['recommendation'])
            '✅ HIGH CONFIDENCE - Bet on Home Win'
        """
        # Create temporary DataFrame
        match_data = pd.DataFrame([{
            'HomeElo': home_elo,
            'AwayElo': away_elo,
            'Form3Home': form3_home,
            'Form5Home': form5_home,
            'Form3Away': form3_away,
            'Form5Away': form5_away
        }])
        
        prob = self.predict_proba(match_data)[0]
        elo_diff = home_elo - away_elo
        form_diff = form5_home - form5_away
        
        # Determine confidence level and recommendation
        if prob >= self.CONFIDENCE_THRESHOLDS['high']:
            confidence = "HIGH"
            recommendation = "✅ HIGH CONFIDENCE - Bet on Home Win"
        elif prob >= self.CONFIDENCE_THRESHOLDS['medium']:
            confidence = "MEDIUM"
            recommendation = "⚠️ MEDIUM CONFIDENCE - Home win likely but risky"
        elif prob >= self.CONFIDENCE_THRESHOLDS['low']:
            confidence = "LOW"
            recommendation = "❌ LOW CONFIDENCE - Match uncertain, avoid betting"
        else:
            confidence = "INVERSE"
            recommendation = f"🔄 INVERSE - Away/Draw likely ({1-prob:.1%})"
        
        return {
            'prob_home_win': float(prob),
            'prob_not_home_win': float(1 - prob),
            'prediction': 'HOME WIN' if prob >= 0.5 else 'NO HOME WIN',
            'confidence': confidence,
            'recommendation': recommendation,
            'elo_diff': float(elo_diff),
            'form_diff': float(form_diff),
            'home_elo': home_elo,
            'away_elo': away_elo
        }
    
    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = 'FTResult'
    ) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            df: Test DataFrame with features and results
            target_col: Column containing actual results
            
        Returns:
            Dictionary with evaluation metrics
        """
        df_prep = self._prepare_features(df)
        df_prep['HomeWin'] = (df_prep[target_col] == 'H').astype(int)
        df_clean = df_prep.dropna(subset=self.feature_names + ['HomeWin'])
        
        y_true = df_clean['HomeWin'].values
        y_pred = self.predict(df_clean)
        y_prob = self.predict_proba(df_clean)
        
        # Confusion matrix components
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        return {
            'n_samples': len(y_true),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'roc_auc': float(roc_auc_score(y_true, y_prob)),
            'brier_score': float(brier_score_loss(y_true, y_prob)),
            'log_loss': float(log_loss(y_true, y_prob)),
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            'confusion_matrix': {
                'true_positive': int(tp),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn)
            }
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance as a sorted DataFrame.
        
        Returns:
            DataFrame with feature names and coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        })
        
        return importance.sort_values('abs_coefficient', ascending=False)
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model (.pkl extension recommended)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'version': '1.0.0'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EPLHomeWinPredictor':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Loaded EPLHomeWinPredictor instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        predictor.training_stats = model_data.get('training_stats', {})
        predictor.is_fitted = True
        
        print(f"✓ Model loaded from: {filepath}")
        return predictor
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        n_features = len(self.feature_names)
        return f"EPLHomeWinPredictor({status}, {n_features} features)"


# Convenience functions for module-level access
def load_model(filepath: str) -> EPLHomeWinPredictor:
    """Load a pre-trained model from disk."""
    return EPLHomeWinPredictor.load(filepath)


def quick_predict(
    home_elo: float,
    away_elo: float,
    model_path: str = 'models/epl_home_win_model.pkl'
) -> Dict:
    """
    Quick prediction using a pre-trained model.
    
    Args:
        home_elo: Home team Elo rating
        away_elo: Away team Elo rating
        model_path: Path to saved model
        
    Returns:
        Prediction dictionary
    """
    predictor = load_model(model_path)
    return predictor.predict_match(home_elo, away_elo)


if __name__ == "__main__":
    # Demo usage
    print("EPL Home Win Predictor - Demo")
    print("="*50)
    
    # Example: Create and demonstrate the predictor
    predictor = EPLHomeWinPredictor()
    print(f"Initialized: {predictor}")
    print(f"Features: {predictor.feature_names}")
