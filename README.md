# Sequential A/B Testing Simulator

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

## Key Findings

This simulator demonstrates that with daily peeking over a 30-day experiment:
- **Expected False Positive Rate**: 5%
- **Actual False Positive Rate**: ~25-30%
- **With O'Brien-Fleming Correction**: ~5-7%

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
