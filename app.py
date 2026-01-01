"""
Sequential A/B Testing Simulator
================================

A Streamlit application demonstrating the dangers of "peeking" at A/B test
results and proper sequential testing corrections.

Author: [Your Name]
Portfolio: [Your Link]
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict

# Import custom modules
from src.simulation_engine import ABTestSimulator, SimulationConfig, calculate_false_positive_rate
from src.statistical_tests import (
    naive_ttest,
    SPRTTest, SPRTConfig,
    AlphaSpendingTest,
    create_spending_test_function,
    obrien_fleming_boundary
)
from src.visualizations import (
    plot_false_positive_comparison,
    plot_peeking_impact,
    plot_alpha_spending_functions,
    plot_cumulative_significance,
    plot_single_experiment_trajectory,
    create_summary_metrics_display,
    create_color_palette
)
from src.utils import format_percentage, get_interpretation, EXPLANATIONS


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Sequential A/B Testing Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .danger-card {
        border-left-color: #EF4444;
    }
    .success-card {
        border-left-color: #22C55E;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #F1F5F9;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

def render_sidebar() -> Dict:
    """Render sidebar and return configuration."""

    st.sidebar.markdown("## Simulation Settings")

    with st.sidebar.expander("Basic Parameters", expanded=True):
        n_simulations = st.slider(
            "Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="More simulations = more accurate estimates, but slower"
        )

        samples_per_day = st.slider(
            "Samples per Day",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Daily traffic to each variant"
        )

        total_days = st.slider(
            "Experiment Duration (days)",
            min_value=7,
            max_value=60,
            value=30,
            step=1
        )

    with st.sidebar.expander("Peeking Behavior", expanded=True):
        peek_frequency = st.slider(
            "Check Results Every N Days",
            min_value=1,
            max_value=7,
            value=1,
            help="1 = daily peeking (worst), 7 = weekly (better)"
        )

    with st.sidebar.expander("Statistical Parameters", expanded=False):
        alpha = st.slider(
            "Significance Level (α)",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            format="%.2f"
        )

        effect_size = st.slider(
            "Effect Size for SPRT (Cohen's d)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum detectable effect for SPRT"
        )

    return {
        'n_simulations': n_simulations,
        'samples_per_day': samples_per_day,
        'total_days': total_days,
        'peek_frequency': peek_frequency,
        'alpha': alpha,
        'effect_size': effect_size
    }


# =============================================================================
# MAIN SIMULATION LOGIC
# =============================================================================

@st.cache_data(show_spinner=False)
def run_comparison_simulation(
    n_simulations: int,
    samples_per_day: int,
    total_days: int,
    peek_frequency: int,
    alpha: float,
    effect_size: float
) -> Dict:
    """
    Run simulations comparing naive and corrected approaches.
    Cached for performance.
    """
    config = SimulationConfig(
        n_simulations=n_simulations,
        samples_per_day=samples_per_day,
        total_days=total_days,
        alpha=alpha,
        true_effect=0.0  # A/A test (null is true)
    )

    total_samples = samples_per_day * 2 * total_days

    results = {}

    # 1. Naive T-test with peeking
    simulator = ABTestSimulator(config)
    naive_results = simulator.run_simulation_batch(
        naive_ttest,
        peek_every=peek_frequency,
        n_simulations=n_simulations
    )
    results['naive'] = calculate_false_positive_rate(naive_results)

    # 2. O'Brien-Fleming
    obf_results_list = []
    for sim_id in range(n_simulations):
        sim = ABTestSimulator(config)
        sim.rng = np.random.default_rng(seed=sim_id)

        tester = AlphaSpendingTest('obrien_fleming', total_samples, alpha)

        control_data = []
        treatment_data = []
        ever_significant = False

        for day in range(1, total_days + 1):
            c, t = sim.generate_daily_data(day, include_effect=False)
            control_data.extend(c)
            treatment_data.extend(t)

            if day % peek_frequency == 0:
                p_val = tester.test(np.array(control_data), np.array(treatment_data))
                if p_val < alpha:
                    ever_significant = True
                    break

        obf_results_list.append({'ever_significant': ever_significant})

    results['obrien_fleming'] = pd.DataFrame(obf_results_list)['ever_significant'].mean()

    # 3. No peeking (final analysis only)
    simulator = ABTestSimulator(config)
    final_only_results = simulator.run_simulation_batch(
        naive_ttest,
        peek_every=total_days,  # Only check at the end
        n_simulations=n_simulations
    )
    results['no_peeking'] = calculate_false_positive_rate(final_only_results)

    # 4. Peeking frequency analysis
    peek_freqs = [1, 2, 3, 5, 7, 10, 15]
    peek_freqs = [p for p in peek_freqs if p <= total_days]
    peek_fprs = []

    for pf in peek_freqs:
        simulator = ABTestSimulator(config)
        pf_results = simulator.run_simulation_batch(
            naive_ttest,
            peek_every=pf,
            n_simulations=min(n_simulations, 2000)  # Faster for this analysis
        )
        peek_fprs.append(calculate_false_positive_rate(pf_results))

    results['peek_analysis'] = {
        'frequencies': peek_freqs,
        'fprs': peek_fprs
    }

    return results


# =============================================================================
# TAB CONTENT
# =============================================================================

def render_problem_tab(results: Dict, config: Dict):
    """Render the 'The Problem' tab."""

    st.markdown("## The Peeking Problem")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Expected False Positive Rate",
            value=format_percentage(config['alpha']),
            delta=None,
            help="What you think your error rate is"
        )

    with col2:
        inflation = results['naive'] / config['alpha']
        st.metric(
            label="Actual Rate (Naive Peeking)",
            value=format_percentage(results['naive']),
            delta=f"+{format_percentage(results['naive'] - config['alpha'])}",
            delta_color="inverse",
            help="What your error rate actually is"
        )

    with col3:
        st.metric(
            label="Inflation Factor",
            value=f"{inflation:.1f}x",
            delta="higher than expected",
            delta_color="inverse"
        )

    st.markdown("---")

    # Peeking impact visualization
    peek_data = results['peek_analysis']
    fig = plot_peeking_impact(
        peek_data['frequencies'],
        peek_data['fprs'],
        config['alpha']
    )
    st.plotly_chart(fig, use_container_width=True)

    # Explanation
    with st.expander("Why does this happen?", expanded=True):
        st.markdown(EXPLANATIONS['peeking_problem'])

        st.markdown("""
        **The math:** Each time you peek, you're essentially running a new statistical
        test. If you peek 30 times at α=0.05, your effective α becomes:

        > 1 - (1 - 0.05)^30 ≈ 0.79

        That's a 79% chance of a false positive, not 5%!
        """)


def render_solutions_tab(results: Dict, config: Dict):
    """Render the 'Solutions' tab."""

    st.markdown("## Sequential Testing Solutions")

    # Comparison chart
    comparison_data = {
        'No Peeking': results['no_peeking'],
        'Naive Peeking': results['naive'],
        "O'Brien-Fleming": results['obrien_fleming']
    }

    fig = plot_false_positive_comparison(comparison_data, config['alpha'])
    st.plotly_chart(fig, use_container_width=True)

    # Method explanations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Alpha Spending Functions")
        with st.expander("How it works", expanded=True):
            st.markdown(EXPLANATIONS['alpha_spending_explanation'])

        # Alpha spending visualization
        fig_spending = plot_alpha_spending_functions(10, config['alpha'])
        st.plotly_chart(fig_spending, use_container_width=True)

    with col2:
        st.markdown("### SPRT (Sequential Probability Ratio Test)")
        with st.expander("How it works", expanded=True):
            st.markdown(EXPLANATIONS['sprt_explanation'])

        # Key advantages
        st.markdown("""
        **Key Advantages:**
        - Mathematically optimal sample efficiency
        - Can stop early for futility OR success
        - No need to pre-specify number of looks

        **Trade-offs:**
        - Must specify effect size upfront
        - More complex to implement
        - Less familiar to stakeholders
        """)


def render_interactive_tab(config: Dict):
    """Render the interactive single-experiment simulator."""

    st.markdown("## Interactive Experiment Simulator")
    st.markdown("Watch a single A/B test unfold day by day")

    col1, col2 = st.columns([1, 3])

    with col1:
        scenario = st.radio(
            "Scenario",
            ["A/A Test (No Effect)", "A/B Test (Real Effect)"],
            help="A/A simulates when null hypothesis is true"
        )

        true_effect = 0.0 if "A/A" in scenario else 0.3 * 20  # 0.3 Cohen's d * std

        if st.button("Run New Experiment", type="primary"):
            st.session_state['experiment_seed'] = np.random.randint(0, 100000)

    # Initialize seed if not exists
    if 'experiment_seed' not in st.session_state:
        st.session_state['experiment_seed'] = 42

    with col2:
        # Run single experiment
        single_config = SimulationConfig(
            n_simulations=1,
            samples_per_day=config['samples_per_day'],
            total_days=config['total_days'],
            alpha=config['alpha'],
            true_effect=true_effect
        )

        simulator = ABTestSimulator(single_config)
        simulator.rng = np.random.default_rng(seed=st.session_state['experiment_seed'])

        result = simulator.run_single_experiment(
            naive_ttest,
            peek_every=1,
            include_effect=(true_effect != 0)
        )

        # Calculate O'Brien-Fleming boundaries
        total_samples = config['samples_per_day'] * 2 * config['total_days']
        obf_alphas = []
        for ss in result['sample_sizes']:
            info_frac = ss * 2 / total_samples
            obf_alphas.append(obrien_fleming_boundary(info_frac, config['alpha']))

        # Plot trajectory
        fig = plot_single_experiment_trajectory(
            result['p_values'],
            result['sample_sizes'],
            config['alpha'],
            obf_alphas
        )
        st.plotly_chart(fig, use_container_width=True)

        # Outcome
        if result['ever_significant']:
            if true_effect == 0:
                st.error(f"**FALSE POSITIVE** - Declared significance on day {result['significant_at_day']}, but there was no real effect.")
            else:
                st.success(f"**TRUE POSITIVE** - Correctly detected effect on day {result['significant_at_day']}.")
        else:
            if true_effect == 0:
                st.success("**CORRECT** - No significance declared (and there was no real effect).")
            else:
                st.warning("**FALSE NEGATIVE** - Missed the real effect (need more power).")


def render_business_impact_tab(results: Dict, config: Dict):
    """Render business impact analysis."""

    st.markdown("## Business Impact Calculator")

    col1, col2 = st.columns(2)

    with col1:
        tests_per_year = st.number_input(
            "A/B Tests Run Per Year",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )

        cost_per_false_positive = st.number_input(
            "Cost of Shipping a Bad Feature ($)",
            min_value=1000,
            max_value=1000000,
            value=50000,
            step=5000,
            help="Engineering time + opportunity cost + potential revenue impact"
        )

    with col2:
        # Calculate impacts
        expected_fp = tests_per_year * config['alpha']
        naive_fp = tests_per_year * results['naive']
        corrected_fp = tests_per_year * results['obrien_fleming']

        wasted_with_naive = (naive_fp - expected_fp) * cost_per_false_positive
        saved_with_correction = (naive_fp - corrected_fp) * cost_per_false_positive

        st.markdown("### Annual Impact")

        st.metric(
            "Expected False Positives (α=0.05)",
            f"{expected_fp:.0f} per year"
        )

        st.metric(
            "Actual False Positives (Naive Peeking)",
            f"{naive_fp:.0f} per year",
            delta=f"+{naive_fp - expected_fp:.0f}"
        )

        st.metric(
            "False Positives (O'Brien-Fleming)",
            f"{corrected_fp:.0f} per year",
            delta=f"{corrected_fp - expected_fp:+.0f}"
        )

    st.markdown("---")

    # Summary
    st.markdown(f"""
    ### Summary

    | Scenario | False Positives/Year | Annual Cost |
    |----------|---------------------|-------------|
    | **Expected** | {expected_fp:.0f} | ${expected_fp * cost_per_false_positive:,.0f} |
    | **Naive Peeking** | {naive_fp:.0f} | ${naive_fp * cost_per_false_positive:,.0f} |
    | **O'Brien-Fleming** | {corrected_fp:.0f} | ${corrected_fp * cost_per_false_positive:,.0f} |

    **Key Finding:** Implementing proper sequential testing would save approximately
    ${saved_with_correction:,.0f} per year by preventing {naive_fp - corrected_fp:.0f}
    false discoveries.
    """)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""

    # Header
    st.markdown('<p class="main-header">Sequential A/B Testing Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Demonstrating why "peeking" breaks statistics and how to fix it</p>', unsafe_allow_html=True)

    # Sidebar configuration
    config = render_sidebar()

    # Run simulations
    with st.spinner("Running simulations... This may take a moment."):
        results = run_comparison_simulation(
            config['n_simulations'],
            config['samples_per_day'],
            config['total_days'],
            config['peek_frequency'],
            config['alpha'],
            config['effect_size']
        )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "The Problem",
        "Solutions",
        "Interactive Demo",
        "Business Impact"
    ])

    with tab1:
        render_problem_tab(results, config)

    with tab2:
        render_solutions_tab(results, config)

    with tab3:
        render_interactive_tab(config)

    with tab4:
        render_business_impact_tab(results, config)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748B; padding: 2rem;'>
        <p>Sequential Testing Simulator | Experimentation Platform</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
