"""
Sequential A/B Testing Simulator - Source Package
"""

from .simulation_engine import ABTestSimulator, SimulationConfig
from .statistical_tests import naive_ttest, SPRTTest, AlphaSpendingTest
from .visualizations import (
    plot_false_positive_comparison,
    plot_peeking_impact,
    plot_alpha_spending_functions
)
