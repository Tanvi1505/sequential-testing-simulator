# Build Notes - Sequential A/B Testing Simulator

## What This Project Does
Interactive Streamlit tool demonstrating Type I error inflation in A/B testing and sequential testing corrections.

## Key Implementation Details

### Statistical Methods
- **Naive t-test**: Standard approach (breaks with peeking)
- **O'Brien-Fleming**: Alpha spending with conservative early boundaries
- **Pocock**: More uniform alpha spending
- **SPRT**: Optimal sequential testing (requires pre-specified effect size)

### Data Generation
- Monte Carlo simulation using NumPy random.normal()
- Default: mean=100, std=20, 100 samples/day
- A/A tests (true_effect=0) to measure false positive rates
- 10,000 simulations for stable estimates

### Architecture
```
app.py                      - Main Streamlit UI (4 tabs)
src/simulation_engine.py    - Monte Carlo simulation logic
src/statistical_tests.py    - All statistical methods
src/visualizations.py       - Plotly charts
src/utils.py               - Helper functions
```

### Key Changes Made
1. Removed all emojis and portfolio branding
2. Added Results table with reproducible numbers
3. Increased chart font sizes (presentation-ready)
4. Added scientific notation to p-value charts
5. Professional footer and messaging

### Deployment
- **GitHub**: https://github.com/Tanvi1505/sequential-testing-simulator
- **Streamlit Cloud**: [URL once deployed]
- **Requirements**: Flexible versions (>=) for cloud compatibility

### Results Summary (Default Config)
| Method | Peek Frequency | False Positive Rate |
|--------|---------------|---------------------|
| Naive | Daily (30 peeks) | 28.5% |
| O'Brien-Fleming | Daily | 5.8% |
| No peeking | Final only | 5.1% |

### If Extending to Real Data
Would need to add:
- Data connectors (SQL/API/CSV)
- Experiment tracking (store interim analyses)
- Decision engine (stop/continue logic)
- Real-time monitoring

Statistical methods in `src/statistical_tests.py` are production-ready and reusable.

### Interview Talking Points
- Why peeking breaks statistics (multiple testing problem)
- How O'Brien-Fleming spending works mathematically
- Why synthetic data (ground truth, reproducibility, scale)
- How to extend for production use
- Trade-offs between methods (SPRT vs alpha spending)

### Common Issues & Fixes
- **Streamlit Cloud deployment error**: Use flexible package versions (>=)
- **Import errors**: Ensure src/__init__.py exists
- **Slow simulations**: Reduce n_simulations or use caching
- **Chart rendering**: Check Plotly version compatibility

### Future Enhancements
- Add Bayesian sequential testing
- Power analysis calculator
- Multiple metrics handling
- Sample size calculator
- Export simulation results to CSV

---

**Built:** January 2026
**Tech Stack:** Python, Streamlit, NumPy, SciPy, Plotly
**Purpose:** Portfolio project demonstrating statistical rigor and software engineering
