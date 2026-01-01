[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sequential-testing-simulator-q4gwd9yswydt7jga9rnq4v.streamlit.app/)

# Sequential A/B Testing Simulator

**Live Demo:** [Try it here](https://sequential-testing-simulator-q4gwd9yswydt7jga9rnq4v.streamlit.app/) | **GitHub:** [View Source](https://github.com/Tanvi1505/sequential-testing-simulator)

## Overview

An interactive Streamlit application demonstrating:
1. **The Peeking Problem**: Why checking A/B test results repeatedly inflates false positive rates
2. **Sequential Testing Solutions**: O'Brien-Fleming, Pocock, and SPRT methods
3. **Business Impact**: Quantifying the cost of statistical mistakes

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Results

Performance of different testing methods under continuous monitoring (10,000 simulations each).

**Simulation Parameters:**
- Sample size: 100 per variant per day
- Duration: 30 days
- Total samples: 3,000 per variant
- Significance level: α = 0.05
- True effect: 0 (A/A test, null hypothesis is true)

| Method | Peek Frequency | False Positive Rate | Avg. Samples to Stop* | Notes |
|--------|---------------|---------------------|---------------------|-------|
| **Naive t-test** | Every day (30 peeks) | 28.5% | ~1,500 | 5.7x inflation |
| **Naive t-test** | Every 3 days (10 peeks) | 16.2% | ~1,800 | 3.2x inflation |
| **Naive t-test** | Every 7 days (4 peeks) | 9.8% | ~2,100 | 2.0x inflation |
| **No peeking** | Final analysis only | 5.1% | 3,000 | Baseline |
| **O'Brien-Fleming** | Every day (30 peeks) | 5.8% | ~2,400 | Controls Type I error |
| **Pocock** | Every day (30 peeks) | 6.4% | ~2,200 | Slightly liberal early |
| **SPRT** | Continuous | 5.2% | ~2,100** | Optimal efficiency |

\* Average sample size when test stops (for tests that stopped early)
\** SPRT with effect size δ = 0.5 Cohen's d, power = 0.80

**Key Takeaway:** Daily peeking with naive testing inflates false positive rate from 5% to 28.5%. Sequential methods maintain proper error control while enabling early stopping.

## Technical Concepts

### Alpha Spending Functions
- **O'Brien-Fleming**: Conservative early, spends most alpha at final analysis
- **Pocock**: Uniform spending across all interim analyses
- **Linear**: Simple proportional spending

### SPRT (Sequential Probability Ratio Test)
- Optimal sample efficiency
- Built for continuous monitoring
- Requires pre-specified effect size


## References I have used to learn:

1. Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials.
2. O'Brien, P. C., & Fleming, T. R. (1979). A multiple testing procedure for clinical trials.
3. Wald, A. (1945). Sequential tests of statistical hypotheses.


STILL WORKING TO FIND A REAL-TIME DATASET TO TEST, HOWEVER THIS CODE IS REUSABLE FOR THE SIMULATION.
