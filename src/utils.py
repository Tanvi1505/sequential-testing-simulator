"""
Utility functions and helper classes.
"""

from typing import Dict, Any
import json


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 0) -> str:
    """Format a number with thousand separators."""
    if decimals == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def get_interpretation(naive_fpr: float, corrected_fpr: float, alpha: float = 0.05) -> Dict[str, str]:
    """
    Generate interpretation text based on results.
    """
    inflation_factor = naive_fpr / alpha

    return {
        'headline': f"Peeking inflated your false positive rate by {inflation_factor:.1f}x",
        'naive_interpretation': (
            f"With naive peeking, you'd incorrectly declare winners "
            f"{format_percentage(naive_fpr)} of the time instead of the expected "
            f"{format_percentage(alpha)}."
        ),
        'corrected_interpretation': (
            f"With proper sequential testing, the false positive rate is "
            f"{format_percentage(corrected_fpr)}, much closer to the target "
            f"{format_percentage(alpha)}."
        ),
        'business_impact': (
            f"In a company running 100 A/A tests per year, naive peeking would "
            f"produce ~{int(naive_fpr * 100)} false discoveries instead of ~{int(alpha * 100)}. "
            f"That's {int((naive_fpr - alpha) * 100)} wasted engineering cycles per year."
        )
    }


# Explanatory content for the app
EXPLANATIONS = {
    'peeking_problem': """
    ### The Peeking Problem

    When you check your A/B test results repeatedly and stop as soon as you see
    "statistical significance," you're not actually controlling your error rate
    at 5%. Each peek is like buying another lottery ticket - more chances to
    see a false positive.

    **Real-world example:** A company runs 100 A/A tests (where there's no real
    difference). With proper testing, ~5 would falsely show significance. With
    daily peeking, that number jumps to 20-30+.
    """,

    'sprt_explanation': """
    ### Sequential Probability Ratio Test (SPRT)

    SPRT flips the script: instead of asking "is this p-value significant?", it
    asks "how much more likely is the data under H1 vs H0?"

    **Key insight:** SPRT was designed from the ground up for sequential analysis.
    It has mathematically guaranteed error rates even with continuous monitoring.

    **Trade-off:** You must pre-specify an effect size you care about detecting.
    """,

    'alpha_spending_explanation': """
    ### Alpha Spending Functions

    Think of your 5% significance level as a "budget" you can spend over multiple
    looks at the data. Alpha spending functions define *how* to spend this budget.

    - **O'Brien-Fleming:** Very conservative early (tiny alpha), spends most at the end
    - **Pocock:** Spends more evenly across all looks
    - **Linear:** Simple proportional spending

    **When to use:** O'Brien-Fleming is preferred in most cases because it preserves
    nearly full power at the final analysis.
    """
}
