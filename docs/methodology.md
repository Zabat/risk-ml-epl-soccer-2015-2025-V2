# Methodology

This document explains the technical approach used in the EPL Home Win Predictor.

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Data Description](#data-description)
3. [Feature Engineering](#feature-engineering)
4. [Model Selection](#model-selection)
5. [Betting Strategy](#betting-strategy)
6. [Risk Management](#risk-management)
7. [Evaluation Metrics](#evaluation-metrics)

---

## Problem Definition

**Objective**: Predict the probability of a home team winning in English Premier League matches.

**Why Home Wins?**
- Home advantage is a well-documented phenomenon in football
- Binary classification is more robust than 3-way (H/D/A)
- Higher baseline accuracy (≈46% home wins in EPL)
- Better suited for betting applications

**Key Insight**: We focus on *high-confidence* predictions where the model has a significant edge, rather than predicting every match.

---

## Data Description

### Dataset
- **Source**: Historical EPL matches (2015-2025)
- **Size**: 3,800 matches
- **Features**: 49 columns including Elo ratings, form, and odds

### Key Variables

| Variable | Description | Type |
|----------|-------------|------|
| `HomeElo` | Elo rating of home team | Continuous (1400-2100) |
| `AwayElo` | Elo rating of away team | Continuous |
| `Form3Home` | Points from last 3 home matches | Discrete (0-9) |
| `Form5Home` | Points from last 5 matches | Discrete (0-15) |
| `Form3Away` | Points from last 3 away matches | Discrete (0-9) |
| `Form5Away` | Points from last 5 matches | Discrete (0-15) |
| `OddHome` | Bookmaker odds for home win | Continuous |
| `FTResult` | Full-time result (H/D/A) | Categorical |

### Elo Ratings

Elo ratings provide a measure of team strength that:
- Updates after each match based on result and opponent strength
- Is self-correcting over time
- Accounts for home advantage (≈65 Elo points)

**Interpretation**:
- 1500: Average team
- 1700: Strong mid-table team
- 1900+: Title contender (Man City, Liverpool)
- <1500: Relegation candidate

---

## Feature Engineering

### Derived Features

```python
# Elo-based features
EloDiff = HomeElo - AwayElo  # Primary predictor
EloSum = HomeElo + AwayElo   # Match quality proxy

# Form-based features
FormDiff = Form5Home - Form5Away  # Recent form comparison
```

### Feature Importance

| Feature | Coefficient | Impact |
|---------|-------------|--------|
| EloDiff | +0.863 | **Dominant** - Elo difference is the strongest predictor |
| Form3Away | -0.118 | Away form negatively impacts home win probability |
| EloSum | +0.094 | Higher quality matches favor home team slightly |
| FormDiff | -0.033 | Recent form has marginal additional value |

### Why These Features?

1. **EloDiff** captures long-term team quality difference
2. **Form variables** capture short-term momentum
3. **EloSum** controls for match importance/quality
4. Simple features reduce overfitting risk

---

## Model Selection

### Why Logistic Regression?

| Model | Pros | Cons |
|-------|------|------|
| **Logistic Regression** ✓ | Interpretable, well-calibrated, fast | Less flexible |
| Random Forest | Handles non-linearity | Overfits, poor calibration |
| Gradient Boosting | High accuracy | Black box, needs tuning |
| Neural Networks | Flexible | Requires more data |

**Key reasons for Logistic Regression**:

1. **Calibration**: Outputs are true probabilities (crucial for Kelly betting)
2. **Interpretability**: Clear feature weights
3. **Robustness**: Less prone to overfitting on limited data
4. **Speed**: Fast training and inference

### Model Specification

```
P(HomeWin) = σ(β₀ + β₁·EloDiff + β₂·EloSum + β₃·Form3Home + ...)

where σ(x) = 1 / (1 + e^(-x))
```

### Training Procedure

1. **Walk-forward validation**: Train on past data, test on future
2. **Retraining**: Model updated every 50 matches
3. **No data leakage**: Only information available at match time is used

---

## Betting Strategy

### Selection Criteria

We only bet when ALL conditions are met:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Predicted probability | ≥ 68% | High confidence only |
| Expected value | ≥ 5% | Positive edge required |
| Odds range | 1.15 - 1.55 | Avoid extreme favorites/underdogs |
| Elo difference | ≥ 250 points | Strong home favorite |

### Why So Selective?

- **39 bets over 10 years** (from 3,800 matches)
- Focus on highest-confidence opportunities
- Bookmakers are efficient on average; edges are rare
- Quality over quantity

### Expected Value Calculation

```
Expected Value = Probability × Odds - 1

Example:
  P(HomeWin) = 75%
  Odds = 1.40
  EV = 0.75 × 1.40 - 1 = +5%
```

---

## Risk Management

### Kelly Criterion

The Kelly Criterion determines optimal bet sizing to maximize long-term growth:

```
f* = (p × b - q) / b

where:
  f* = fraction of bankroll to bet
  p = probability of winning
  q = probability of losing (1-p)
  b = net odds (decimal odds - 1)
```

### Fractional Kelly

Full Kelly is theoretically optimal but has high variance. We recommend:

| Strategy | Fraction | Return | Drawdown | Use Case |
|----------|----------|--------|----------|----------|
| Full Kelly | 100% | Highest | 30%+ | Aggressive |
| Half Kelly | 50% | Moderate | 15-20% | Balanced |
| **Quarter Kelly** | 25% | Good | 10-15% | **Recommended** |
| Eighth Kelly | 12.5% | Lower | <10% | Conservative |

### Dynamic Leverage

Adjust bet sizing based on current capital:

| Capital Level | Strategy | Leverage |
|--------------|----------|----------|
| < 50% of initial | Aggressive recovery | 2-3x |
| 50-100% | Recovery mode | 1.5-2x |
| 100-150% | Growth mode | 1x |
| > 150% | Protection mode | 0.5x |

---

## Evaluation Metrics

### Prediction Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 66.6% | Correct predictions |
| ROC-AUC | 0.72 | Discrimination ability |
| Brier Score | 0.21 | Calibration quality (lower is better) |
| Log Loss | 0.58 | Probabilistic accuracy |

### Betting Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Win Rate | 79.5% | % of winning bets |
| ROI | +12.8% | Return on investment |
| Sharpe Ratio | 1.62 | Risk-adjusted return |
| Max Drawdown | 17.3% | Largest peak-to-trough decline |

### Calibration

A well-calibrated model means:
- When we predict 70%, the event occurs ≈70% of the time
- Critical for Kelly Criterion to work properly

Our model shows excellent calibration in the 70-90% probability range.

---

## Limitations

1. **Sample size**: Only 39 bets over 10 years limits statistical significance
2. **Market efficiency**: Edges may disappear as markets become more efficient
3. **Historical data**: Past performance doesn't guarantee future results
4. **Odds availability**: Model assumes odds were available at predicted levels
5. **Transaction costs**: Doesn't account for betting taxes/fees

---

## References

1. Elo, A. (1978). *The Rating of Chess Players, Past and Present*
2. Kelly, J.L. (1956). "A New Interpretation of Information Rate"
3. Kovalchik, S. (2016). "Searching for the GOAT of tennis win prediction"
4. Hvattum, L.M. & Arntzen, H. (2010). "Using ELO ratings for match result prediction"
