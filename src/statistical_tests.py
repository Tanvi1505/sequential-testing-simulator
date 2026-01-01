"""
Statistical testing functions including:
- Standard T-test (naive approach)
- Sequential Probability Ratio Test (SPRT)
- O'Brien-Fleming Alpha Spending
- Pocock Alpha Spending
"""

import numpy as np
from scipy import stats
from scipy.special import ndtri  # Inverse normal CDF (same as norm.ppf)
from typing import Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# NAIVE T-TEST (The Problem)
# =============================================================================

def naive_ttest(control: np.ndarray, treatment: np.ndarray) -> float:
    """
    Standard two-sample t-test. Returns p-value.
    This is the "naive" approach that inflates false positive rate when peeking.
    """
    # Handle edge case of insufficient data
    if len(control) < 2 or len(treatment) < 2:
        return 1.0

    _, p_value = stats.ttest_ind(control, treatment, equal_var=False)
    return p_value


# =============================================================================
# SEQUENTIAL PROBABILITY RATIO TEST (SPRT)
# =============================================================================

@dataclass
class SPRTConfig:
    """Configuration for SPRT."""
    alpha: float = 0.05  # Type I error rate
    beta: float = 0.20   # Type II error rate (1 - power)
    delta: float = 0.5   # Effect size (Cohen's d) to detect


class SPRTTest:
    """
    Sequential Probability Ratio Test for continuous data.

    Uses Wald's SPRT with boundaries based on alpha and beta.
    """

    def __init__(self, config: SPRTConfig):
        self.config = config
        # Calculate log boundaries
        self.upper_boundary = np.log((1 - config.beta) / config.alpha)
        self.lower_boundary = np.log(config.beta / (1 - config.alpha))

    def calculate_log_likelihood_ratio(
        self,
        control: np.ndarray,
        treatment: np.ndarray
    ) -> float:
        """
        Calculate the log likelihood ratio for the current data.

        Under H0: difference = 0
        Under H1: difference = delta * pooled_std
        """
        if len(control) < 2 or len(treatment) < 2:
            return 0.0

        # Calculate statistics
        n_c, n_t = len(control), len(treatment)
        mean_c, mean_t = np.mean(control), np.mean(treatment)
        var_c, var_t = np.var(control, ddof=1), np.var(treatment, ddof=1)

        # Pooled standard deviation
        pooled_var = ((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2)
        pooled_std = np.sqrt(pooled_var)

        if pooled_std == 0:
            return 0.0

        # Observed effect size
        observed_diff = mean_t - mean_c

        # Standard error of the difference
        se_diff = pooled_std * np.sqrt(1/n_c + 1/n_t)

        # Effect size under H1
        effect_under_h1 = self.config.delta * pooled_std

        # Log likelihood ratio (Wald's approximation for normal data)
        # LLR = (observed - h0) * h1 / variance - 0.5 * (h1^2 - h0^2) / variance
        llr = (observed_diff * effect_under_h1 / (se_diff ** 2)) - \
              (0.5 * (effect_under_h1 ** 2) / (se_diff ** 2))

        return llr

    def get_decision(
        self,
        control: np.ndarray,
        treatment: np.ndarray
    ) -> Tuple[str, float]:
        """
        Make a decision based on SPRT.

        Returns:
            Tuple of (decision, log_likelihood_ratio)
            decision is one of: 'reject_null', 'accept_null', 'continue'
        """
        llr = self.calculate_log_likelihood_ratio(control, treatment)

        if llr >= self.upper_boundary:
            return 'reject_null', llr
        elif llr <= self.lower_boundary:
            return 'accept_null', llr
        else:
            return 'continue', llr

    def as_test_function(self, control: np.ndarray, treatment: np.ndarray) -> float:
        """
        Convert to p-value-like output for compatibility with simulator.
        Returns 0 if reject null, 1 if accept null, 0.5 if continue.
        """
        decision, _ = self.get_decision(control, treatment)
        if decision == 'reject_null':
            return 0.001  # Very significant
        elif decision == 'accept_null':
            return 0.999  # Not significant
        else:
            return 0.5  # Inconclusive


# =============================================================================
# ALPHA SPENDING FUNCTIONS
# =============================================================================

def obrien_fleming_boundary(information_fraction: float, alpha: float = 0.05) -> float:
    """
    Calculate O'Brien-Fleming adjusted critical value.

    The O'Brien-Fleming spending function is:
    α(t) = 2 - 2 * Φ(z_{α/2} / sqrt(t))

    where t is the information fraction (0 to 1).

    Args:
        information_fraction: Proportion of total planned samples collected (0 to 1)
        alpha: Overall significance level

    Returns:
        Adjusted alpha level for this interim analysis
    """
    if information_fraction <= 0:
        return 0.0
    if information_fraction >= 1:
        return alpha

    # z-score for alpha/2 (two-sided test)
    z_alpha = ndtri(1 - alpha / 2)

    # O'Brien-Fleming spending function (Lan-DeMets approximation)
    spent_alpha = 2 * (1 - stats.norm.cdf(z_alpha / np.sqrt(information_fraction)))

    return min(spent_alpha, alpha)


def pocock_boundary(information_fraction: float, alpha: float = 0.05) -> float:
    """
    Calculate Pocock adjusted critical value.

    The Pocock spending function is:
    α(t) = α * ln(1 + (e-1) * t)

    Args:
        information_fraction: Proportion of total planned samples collected (0 to 1)
        alpha: Overall significance level

    Returns:
        Adjusted alpha level for this interim analysis
    """
    if information_fraction <= 0:
        return 0.0
    if information_fraction >= 1:
        return alpha

    # Pocock spending function
    spent_alpha = alpha * np.log(1 + (np.e - 1) * information_fraction)

    return min(spent_alpha, alpha)


def linear_spending(information_fraction: float, alpha: float = 0.05) -> float:
    """
    Simple linear alpha spending function.
    α(t) = α * t

    This is a compromise between O'Brien-Fleming (conservative early) and
    Pocock (spend evenly).
    """
    return alpha * information_fraction


class AlphaSpendingTest:
    """
    T-test with alpha spending correction for sequential testing.
    """

    def __init__(
        self,
        spending_function: str = 'obrien_fleming',
        total_samples: int = 3000,
        alpha: float = 0.05
    ):
        self.spending_function = spending_function
        self.total_samples = total_samples
        self.alpha = alpha
        self.cumulative_alpha_spent = 0.0
        self.last_alpha_boundary = 0.0

        # Select spending function
        if spending_function == 'obrien_fleming':
            self._spend_func = obrien_fleming_boundary
        elif spending_function == 'pocock':
            self._spend_func = pocock_boundary
        elif spending_function == 'linear':
            self._spend_func = linear_spending
        else:
            raise ValueError(f"Unknown spending function: {spending_function}")

    def get_adjusted_alpha(self, current_samples: int) -> float:
        """
        Get the alpha level to use at this interim analysis.
        """
        info_fraction = current_samples / self.total_samples
        cumulative_alpha = self._spend_func(info_fraction, self.alpha)

        # Incremental alpha for this analysis
        incremental_alpha = cumulative_alpha - self.cumulative_alpha_spent
        self.cumulative_alpha_spent = cumulative_alpha
        self.last_alpha_boundary = max(incremental_alpha, 0.0001)

        return self.last_alpha_boundary

    def test(self, control: np.ndarray, treatment: np.ndarray) -> float:
        """
        Perform t-test with alpha spending adjustment.
        Returns adjusted p-value (actually returns original p-value,
        but significance is determined by comparing to adjusted alpha).
        """
        if len(control) < 2 or len(treatment) < 2:
            return 1.0

        current_samples = len(control) + len(treatment)
        adjusted_alpha = self.get_adjusted_alpha(current_samples)

        _, p_value = stats.ttest_ind(control, treatment, equal_var=False)

        # Return a modified p-value that accounts for the spending
        # If p < adjusted_alpha, return something < 0.05
        # Otherwise, return something > 0.05
        if p_value < adjusted_alpha:
            return p_value * (self.alpha / adjusted_alpha) * 0.5
        else:
            return min(p_value / adjusted_alpha, 1.0)

    def reset(self):
        """Reset the spending tracker for a new experiment."""
        self.cumulative_alpha_spent = 0.0
        self.last_alpha_boundary = 0.0


def create_spending_test_function(
    spending_function: str,
    total_samples: int,
    alpha: float = 0.05
) -> callable:
    """
    Factory function to create a fresh alpha spending test for each simulation.
    """
    def test_func(control: np.ndarray, treatment: np.ndarray) -> float:
        tester = AlphaSpendingTest(spending_function, total_samples, alpha)
        return tester.test(control, treatment)

    return test_func
