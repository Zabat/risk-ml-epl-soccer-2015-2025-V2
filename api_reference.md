# API Reference

Complete API documentation for the EPL Betting Model.

## Table of Contents

- [EPLHomeWinPredictor](#eplhomewinpredictor)
- [BettingBacktest](#bettingbacktest)
- [Risk Analysis Functions](#risk-analysis-functions)
- [Utility Functions](#utility-functions)

---

## EPLHomeWinPredictor

Main prediction model using Logistic Regression.

### Class Definition

```python
class EPLHomeWinPredictor:
    """
    Logistic Regression model for predicting EPL home wins.
    
    Attributes:
        model: Trained LogisticRegression model
        scaler: StandardScaler for feature normalization
        feature_names: List of feature column names
        is_fitted: Whether the model has been trained
        training_stats: Dictionary of training statistics
    """
```

### Constructor

```python
EPLHomeWinPredictor(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    features: Optional[List[str]] = None
)
```

**Parameters:**
- `C`: Inverse regularization strength (smaller = stronger regularization)
- `max_iter`: Maximum iterations for solver
- `random_state`: Random seed for reproducibility
- `features`: Custom feature list (defaults to standard features)

### Methods

#### fit()

```python
def fit(
    df: pd.DataFrame,
    target_col: str = 'FTResult',
    verbose: bool = True
) -> 'EPLHomeWinPredictor'
```

Train the model on historical match data.

**Parameters:**
- `df`: DataFrame with match data
- `target_col`: Column containing match result ('H', 'D', 'A')
- `verbose`: Whether to print training statistics

**Returns:** Fitted predictor instance

**Example:**
```python
predictor = EPLHomeWinPredictor()
predictor.fit(training_data)
```

---

#### predict_proba()

```python
def predict_proba(df: pd.DataFrame) -> np.ndarray
```

Predict home win probability for given matches.

**Parameters:**
- `df`: DataFrame with match features

**Returns:** Array of probabilities (0 to 1)

**Example:**
```python
probabilities = predictor.predict_proba(test_data)
print(f"Home win probability: {probabilities[0]:.1%}")
```

---

#### predict_match()

```python
def predict_match(
    home_elo: float,
    away_elo: float,
    form3_home: float = 4.5,
    form5_home: float = 7.5,
    form3_away: float = 4.5,
    form5_away: float = 7.5
) -> Dict
```

Predict a single match with detailed analysis.

**Parameters:**
- `home_elo`: Elo rating of home team (typically 1400-2100)
- `away_elo`: Elo rating of away team
- `form3_home`: Home team points from last 3 matches (0-9)
- `form5_home`: Home team points from last 5 matches (0-15)
- `form3_away`: Away team points from last 3 matches (0-9)
- `form5_away`: Away team points from last 5 matches (0-15)

**Returns:** Dictionary containing:
```python
{
    'prob_home_win': float,      # Probability of home win
    'prob_not_home_win': float,  # Probability of draw/away
    'prediction': str,           # 'HOME WIN' or 'NO HOME WIN'
    'confidence': str,           # 'HIGH', 'MEDIUM', 'LOW', 'INVERSE'
    'recommendation': str,       # Betting recommendation
    'elo_diff': float,          # Elo difference
    'form_diff': float          # Form difference
}
```

**Example:**
```python
result = predictor.predict_match(
    home_elo=1900,
    away_elo=1700,
    form5_home=10,
    form5_away=6
)
print(result['recommendation'])
# '✅ HIGH CONFIDENCE - Bet on Home Win'
```

---

#### save() / load()

```python
def save(filepath: str) -> None
def load(cls, filepath: str) -> 'EPLHomeWinPredictor'
```

Save and load trained models.

**Example:**
```python
# Save
predictor.save('models/my_model.pkl')

# Load
loaded = EPLHomeWinPredictor.load('models/my_model.pkl')
```

---

## BettingBacktest

Backtesting engine for betting strategies.

### Class Definition

```python
class BettingBacktest:
    """
    Backtesting engine for sports betting strategies.
    
    Attributes:
        initial_bankroll: Starting capital
        bankroll: Current capital
        bankroll_history: List of bankroll values
        bet_history: List of BetRecord objects
    """
```

### Constructor

```python
BettingBacktest(initial_bankroll: float = 1000)
```

### Methods

#### calculate_kelly()

```python
def calculate_kelly(
    prob: float,
    odds: float,
    fraction: float = 0.25
) -> float
```

Calculate optimal bet size using Kelly Criterion.

**Parameters:**
- `prob`: Estimated probability of winning (0 to 1)
- `odds`: Decimal odds
- `fraction`: Kelly fraction (0.25 = quarter Kelly)

**Returns:** Optimal bet fraction of bankroll

**Example:**
```python
backtest = BettingBacktest()
bet_size = backtest.calculate_kelly(prob=0.70, odds=1.50, fraction=0.25)
print(f"Bet {bet_size:.1%} of bankroll")
```

---

#### place_bet()

```python
def place_bet(
    match_info: pd.Series,
    bet_type: str,
    odds: float,
    prob: float,
    bet_fraction: float
) -> bool
```

Execute a bet and update bankroll.

**Parameters:**
- `match_info`: Series with match details
- `bet_type`: 'H' (home), 'D' (draw), or 'A' (away)
- `odds`: Decimal odds
- `prob`: Estimated probability
- `bet_fraction`: Fraction of bankroll to bet

**Returns:** True if bet won, False otherwise

---

#### get_stats()

```python
def get_stats() -> Optional[Dict[str, Any]]
```

Calculate comprehensive performance statistics.

**Returns:** Dictionary with:
```python
{
    'total_bets': int,
    'wins': int,
    'losses': int,
    'win_rate': float,
    'avg_odds': float,
    'total_profit': float,
    'roi': float,
    'final_bankroll': float,
    'max_drawdown': float,
    'sharpe_ratio': float
}
```

---

### run_backtest()

```python
def run_backtest(
    df: pd.DataFrame,
    config: Optional[Dict] = None
) -> Tuple[BettingBacktest, EPLHomeWinPredictor]
```

Execute a complete backtest.

**Parameters:**
- `df`: DataFrame with historical match data
- `config`: Configuration dictionary

**Config Options:**
```python
config = {
    'min_prob': 0.68,        # Minimum predicted probability
    'min_value': 0.05,       # Minimum expected value
    'min_odds': 1.15,        # Minimum odds
    'max_odds': 1.55,        # Maximum odds
    'min_elo_diff': 250,     # Minimum Elo advantage
    'kelly_fraction': 0.25,  # Kelly fraction
    'max_bet': 0.06,         # Maximum bet size
    'initial_bankroll': 1000,
    'train_window': 300,
    'retrain_every': 50
}
```

**Example:**
```python
backtest, predictor = run_backtest(data, config)
print(backtest.get_stats())
```

---

## Risk Analysis Functions

### kelly_fraction()

```python
def kelly_fraction(prob: float, odds: float) -> float
```

Calculate full Kelly fraction.

### calculate_cvar()

```python
def calculate_cvar(
    returns: np.ndarray,
    confidence: float = 0.95
) -> float
```

Calculate Conditional Value at Risk (Expected Shortfall).

**Parameters:**
- `returns`: Array of returns or final values
- `confidence`: Confidence level (0.95 = 95%)

**Returns:** Expected value in worst (1-confidence) cases

### run_monte_carlo()

```python
def run_monte_carlo(
    bet_opportunities: List[Dict],
    n_simulations: int = 1000,
    initial_bankroll: float = 1000,
    kelly_fraction: float = 0.25
) -> Dict
```

Run Monte Carlo simulation of betting strategy.

**Returns:**
```python
{
    'final_bankrolls': np.ndarray,
    'mean': float,
    'median': float,
    'var_95': float,
    'cvar_95': float,
    'prob_profit': float,
    'prob_ruin': float
}
```

---

## Utility Functions

### load_epl_data()

```python
def load_epl_data(filepath: str) -> pd.DataFrame
```

Load and preprocess EPL match data.

### calculate_sharpe_ratio()

```python
def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0,
    periods_per_year: int = 252
) -> float
```

Calculate annualized Sharpe ratio.

### calculate_max_drawdown()

```python
def calculate_max_drawdown(
    values: np.ndarray
) -> Tuple[float, int, int]
```

Calculate maximum drawdown with peak and trough indices.

---

## Quick Reference

### Common Imports

```python
from src import (
    EPLHomeWinPredictor,
    BettingBacktest,
    run_backtest,
    kelly_fraction,
    calculate_cvar,
    run_monte_carlo
)
```

### Typical Workflow

```python
# 1. Load data
import pandas as pd
df = pd.read_csv('data/EPL_League_2015_2025.csv')

# 2. Train model
from src import EPLHomeWinPredictor
predictor = EPLHomeWinPredictor()
predictor.fit(df)

# 3. Make predictions
result = predictor.predict_match(home_elo=1900, away_elo=1700)

# 4. Run backtest
from src import run_backtest
backtest, _ = run_backtest(df, {'kelly_fraction': 0.25})

# 5. Analyze results
stats = backtest.get_stats()
print(f"ROI: {stats['roi']:.1%}")
```
