"""
Plotly visualization functions for the sequential testing simulator.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional


def create_color_palette():
    """Define consistent color palette."""
    return {
        'naive': '#EF4444',      # Red - danger
        'obrien_fleming': '#22C55E',  # Green - safe
        'pocock': '#3B82F6',     # Blue
        'linear': '#F59E0B',     # Orange
        'sprt': '#8B5CF6',       # Purple
        'background': '#F8FAFC',
        'grid': '#E2E8F0',
        'text': '#1E293B'
    }


def plot_false_positive_comparison(results: Dict[str, float], target_alpha: float = 0.05) -> go.Figure:
    """
    Create a bar chart comparing false positive rates across methods.

    Args:
        results: Dict mapping method name to false positive rate
        target_alpha: The nominal alpha level (for reference line)
    """
    colors = create_color_palette()

    methods = list(results.keys())
    rates = list(results.values())

    # Color bars based on whether they exceed target
    bar_colors = [
        colors['naive'] if rate > target_alpha * 1.5 else colors['obrien_fleming']
        for rate in rates
    ]

    fig = go.Figure()

    # Add bars
    fig.add_trace(go.Bar(
        x=methods,
        y=[r * 100 for r in rates],  # Convert to percentage
        marker_color=bar_colors,
        text=[f'{r*100:.1f}%' for r in rates],
        textposition='outside',
        textfont=dict(size=16),
        hovertemplate='%{x}<br>False Positive Rate: %{y:.1f}%<extra></extra>'
    ))

    # Add target line
    fig.add_hline(
        y=target_alpha * 100,
        line_dash='dash',
        line_color=colors['text'],
        annotation_text=f'Target: {target_alpha*100:.0f}%',
        annotation_position='right',
        annotation_font_size=14
    )

    fig.update_layout(
        title={
            'text': 'False Positive Rates by Testing Method',
            'font': {'size': 22, 'color': colors['text']}
        },
        xaxis_title='Testing Method',
        yaxis_title='False Positive Rate (%)',
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=14),
        yaxis_range=[0, max(rates) * 120],
        plot_bgcolor=colors['background'],
        paper_bgcolor='white',
        font={'color': colors['text'], 'size': 14},
        showlegend=False,
        height=500
    )

    fig.update_xaxes(gridcolor=colors['grid'])
    fig.update_yaxes(gridcolor=colors['grid'])

    return fig


def plot_peeking_impact(
    peek_frequencies: List[int],
    false_positive_rates: List[float],
    target_alpha: float = 0.05
) -> go.Figure:
    """
    Show how false positive rate increases with peeking frequency.
    """
    colors = create_color_palette()

    fig = go.Figure()

    # Add the main line
    fig.add_trace(go.Scatter(
        x=peek_frequencies,
        y=[r * 100 for r in false_positive_rates],
        mode='lines+markers',
        marker=dict(size=10, color=colors['naive']),
        line=dict(color=colors['naive'], width=3),
        name='Observed FPR',
        hovertemplate='Peek every %{x} days<br>FPR: %{y:.1f}%<extra></extra>'
    ))

    # Add target line
    fig.add_hline(
        y=target_alpha * 100,
        line_dash='dash',
        line_color=colors['obrien_fleming'],
        annotation_text=f'Expected: {target_alpha*100:.0f}%',
        annotation_position='right'
    )

    # Add danger zone shading
    fig.add_hrect(
        y0=target_alpha * 100,
        y1=max(false_positive_rates) * 110,
        fillcolor=colors['naive'],
        opacity=0.1,
        line_width=0,
        annotation_text='Danger Zone',
        annotation_position='top left'
    )

    fig.update_layout(
        title={
            'text': 'The Cost of Peeking: False Positive Rate vs. Check Frequency',
            'font': {'size': 20, 'color': colors['text']}
        },
        xaxis_title='Check Every N Days',
        yaxis_title='False Positive Rate (%)',
        plot_bgcolor=colors['background'],
        paper_bgcolor='white',
        font={'color': colors['text']},
        height=500
    )

    fig.update_xaxes(gridcolor=colors['grid'])
    fig.update_yaxes(gridcolor=colors['grid'])

    return fig


def plot_cumulative_significance(
    days: List[int],
    naive_rates: List[float],
    corrected_rates: List[float],
    method_name: str = 'O\'Brien-Fleming'
) -> go.Figure:
    """
    Show cumulative probability of reaching significance over time.
    """
    colors = create_color_palette()

    fig = go.Figure()

    # Naive approach
    fig.add_trace(go.Scatter(
        x=days,
        y=[r * 100 for r in naive_rates],
        mode='lines',
        fill='tozeroy',
        name='Naive T-test',
        line=dict(color=colors['naive'], width=2),
        fillcolor=f'rgba(239, 68, 68, 0.2)'
    ))

    # Corrected approach
    fig.add_trace(go.Scatter(
        x=days,
        y=[r * 100 for r in corrected_rates],
        mode='lines',
        fill='tozeroy',
        name=method_name,
        line=dict(color=colors['obrien_fleming'], width=2),
        fillcolor=f'rgba(34, 197, 94, 0.2)'
    ))

    fig.update_layout(
        title={
            'text': 'Cumulative False Positives Over Time',
            'font': {'size': 20, 'color': colors['text']}
        },
        xaxis_title='Day',
        yaxis_title='Cumulative False Positive Rate (%)',
        plot_bgcolor=colors['background'],
        paper_bgcolor='white',
        font={'color': colors['text']},
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01
        ),
        height=500
    )

    fig.update_xaxes(gridcolor=colors['grid'])
    fig.update_yaxes(gridcolor=colors['grid'])

    return fig


def plot_alpha_spending_functions(total_looks: int = 10, alpha: float = 0.05) -> go.Figure:
    """
    Visualize different alpha spending functions.
    """
    from src.statistical_tests import obrien_fleming_boundary, pocock_boundary, linear_spending

    colors = create_color_palette()

    # Generate information fractions
    info_fractions = np.linspace(0.1, 1.0, total_looks)

    # Calculate spent alpha for each function
    obf_spent = [obrien_fleming_boundary(t, alpha) for t in info_fractions]
    pocock_spent = [pocock_boundary(t, alpha) for t in info_fractions]
    linear_spent = [linear_spending(t, alpha) for t in info_fractions]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=info_fractions * 100,
        y=[s * 100 for s in obf_spent],
        mode='lines+markers',
        name="O'Brien-Fleming",
        line=dict(color=colors['obrien_fleming'], width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=info_fractions * 100,
        y=[s * 100 for s in pocock_spent],
        mode='lines+markers',
        name='Pocock',
        line=dict(color=colors['pocock'], width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=info_fractions * 100,
        y=[s * 100 for s in linear_spent],
        mode='lines+markers',
        name='Linear',
        line=dict(color=colors['linear'], width=3),
        marker=dict(size=8)
    ))

    # Add total alpha line
    fig.add_hline(
        y=alpha * 100,
        line_dash='dot',
        line_color=colors['text'],
        annotation_text=f'Total α = {alpha*100:.0f}%'
    )

    fig.update_layout(
        title={
            'text': 'Alpha Spending Functions: How Much "Error Budget" Is Used Over Time',
            'font': {'size': 18, 'color': colors['text']}
        },
        xaxis_title='Information Fraction (%)',
        yaxis_title='Cumulative Alpha Spent (%)',
        plot_bgcolor=colors['background'],
        paper_bgcolor='white',
        font={'color': colors['text']},
        legend=dict(
            yanchor='bottom',
            y=0.01,
            xanchor='right',
            x=0.99
        ),
        height=500
    )

    fig.update_xaxes(gridcolor=colors['grid'])
    fig.update_yaxes(gridcolor=colors['grid'], range=[0, alpha * 120])

    return fig


def plot_single_experiment_trajectory(
    p_values: List[float],
    sample_sizes: List[int],
    alpha: float = 0.05,
    adjusted_alphas: Optional[List[float]] = None
) -> go.Figure:
    """
    Show p-value trajectory for a single experiment with optional adjusted boundaries.
    """
    colors = create_color_palette()

    fig = go.Figure()

    # P-value trajectory
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=p_values,
        mode='lines+markers',
        name='P-value',
        line=dict(color=colors['pocock'], width=2),
        marker=dict(size=6)
    ))

    # Constant alpha line
    fig.add_hline(
        y=alpha,
        line_dash='dash',
        line_color=colors['naive'],
        annotation_text=f'Naive α = {alpha}',
        annotation_position='right'
    )

    # Adjusted alpha boundaries if provided
    if adjusted_alphas:
        fig.add_trace(go.Scatter(
            x=sample_sizes,
            y=adjusted_alphas,
            mode='lines',
            name='Adjusted α (O\'Brien-Fleming)',
            line=dict(color=colors['obrien_fleming'], width=2, dash='dot')
        ))

    fig.update_layout(
        title={
            'text': 'P-Value Trajectory Over Time',
            'font': {'size': 18, 'color': colors['text']}
        },
        xaxis_title='Sample Size',
        yaxis_title='P-Value',
        yaxis_type='log',
        plot_bgcolor=colors['background'],
        paper_bgcolor='white',
        font={'color': colors['text']},
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99
        ),
        height=450
    )

    fig.update_xaxes(gridcolor=colors['grid'])
    fig.update_yaxes(
        gridcolor=colors['grid'],
        tickformat='.0e',
        exponentformat='e'
    )

    return fig


def plot_decision_boundaries_sprt(
    log_likelihood_ratios: List[float],
    upper_bound: float,
    lower_bound: float
) -> go.Figure:
    """
    Visualize SPRT decision process with boundaries.
    """
    colors = create_color_palette()

    observations = list(range(1, len(log_likelihood_ratios) + 1))

    fig = go.Figure()

    # LLR trajectory
    fig.add_trace(go.Scatter(
        x=observations,
        y=log_likelihood_ratios,
        mode='lines+markers',
        name='Log Likelihood Ratio',
        line=dict(color=colors['sprt'], width=2)
    ))

    # Upper boundary (reject H0)
    fig.add_hline(
        y=upper_bound,
        line_dash='dash',
        line_color=colors['obrien_fleming'],
        annotation_text='Reject H₀ (Effect exists)'
    )

    # Lower boundary (accept H0)
    fig.add_hline(
        y=lower_bound,
        line_dash='dash',
        line_color=colors['naive'],
        annotation_text='Accept H₀ (No effect)'
    )

    # Continue zone shading
    fig.add_hrect(
        y0=lower_bound,
        y1=upper_bound,
        fillcolor='gray',
        opacity=0.1,
        annotation_text='Continue Sampling',
        annotation_position='inside'
    )

    fig.update_layout(
        title={
            'text': 'SPRT Decision Boundaries',
            'font': {'size': 18, 'color': colors['text']}
        },
        xaxis_title='Observation Number',
        yaxis_title='Log Likelihood Ratio',
        plot_bgcolor=colors['background'],
        paper_bgcolor='white',
        font={'color': colors['text']},
        height=450
    )

    return fig


def create_summary_metrics_display(
    naive_fpr: float,
    corrected_fpr: float,
    method_name: str,
    target_alpha: float = 0.05
) -> go.Figure:
    """
    Create a visual summary of key metrics.
    """
    colors = create_color_palette()

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=['Target α', 'Naive Approach', f'{method_name}']
    )

    # Target
    fig.add_trace(go.Indicator(
        mode='number',
        value=target_alpha * 100,
        number={'suffix': '%', 'font': {'size': 48, 'color': colors['text']}},
        title={'text': 'Expected', 'font': {'size': 16}}
    ), row=1, col=1)

    # Naive
    fig.add_trace(go.Indicator(
        mode='number+delta',
        value=naive_fpr * 100,
        number={'suffix': '%', 'font': {'size': 48, 'color': colors['naive']}},
        delta={'reference': target_alpha * 100, 'relative': False, 'suffix': '%'},
        title={'text': 'Actual', 'font': {'size': 16}}
    ), row=1, col=2)

    # Corrected
    fig.add_trace(go.Indicator(
        mode='number+delta',
        value=corrected_fpr * 100,
        number={'suffix': '%', 'font': {'size': 48, 'color': colors['obrien_fleming']}},
        delta={'reference': target_alpha * 100, 'relative': False, 'suffix': '%'},
        title={'text': 'Actual', 'font': {'size': 16}}
    ), row=1, col=3)

    fig.update_layout(
        height=250,
        paper_bgcolor='white'
    )

    return fig
