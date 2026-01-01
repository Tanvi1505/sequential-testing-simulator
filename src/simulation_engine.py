"""
Core simulation engine for A/B testing scenarios.
Implements A/A testing (null hypothesis true) and A/B testing (with true effect).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    n_simulations: int = 10000
    samples_per_day: int = 100
    total_days: int = 30
    peek_frequency: int = 1  # Check every N days
    alpha: float = 0.05
    true_effect: float = 0.0  # 0.0 for A/A test, non-zero for A/B
    control_mean: float = 100.0
    control_std: float = 20.0


class ABTestSimulator:
    """
    Simulates A/B tests with configurable peeking behavior.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(seed=42)

    def generate_daily_data(self, day: int, include_effect: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one day's worth of data for control and treatment groups.

        Args:
            day: Current day number (for reproducibility)
            include_effect: Whether to add treatment effect

        Returns:
            Tuple of (control_data, treatment_data)
        """
        n = self.config.samples_per_day

        control = self.rng.normal(
            self.config.control_mean,
            self.config.control_std,
            n
        )

        treatment_mean = self.config.control_mean
        if include_effect:
            treatment_mean += self.config.true_effect

        treatment = self.rng.normal(
            treatment_mean,
            self.config.control_std,
            n
        )

        return control, treatment

    def run_single_experiment(
        self,
        test_function: callable,
        peek_every: int = 1,
        include_effect: bool = False
    ) -> dict:
        """
        Run a single A/B test experiment with peeking.

        Args:
            test_function: Statistical test to use (returns p-value)
            peek_every: How often to check for significance (in days)
            include_effect: Whether this is an A/B test (vs A/A)

        Returns:
            Dict with results including when/if significance was reached
        """
        control_data = []
        treatment_data = []

        p_values = []
        sample_sizes = []
        significant_at_day = None

        for day in range(1, self.config.total_days + 1):
            # Generate and accumulate data
            c, t = self.generate_daily_data(day, include_effect)
            control_data.extend(c)
            treatment_data.extend(t)

            # Check for significance at peek intervals
            if day % peek_every == 0:
                p_value = test_function(
                    np.array(control_data),
                    np.array(treatment_data)
                )
                p_values.append(p_value)
                sample_sizes.append(len(control_data))

                # Record first time significance is reached
                if significant_at_day is None and p_value < self.config.alpha:
                    significant_at_day = day

        return {
            'p_values': p_values,
            'sample_sizes': sample_sizes,
            'significant_at_day': significant_at_day,
            'final_p_value': p_values[-1] if p_values else None,
            'ever_significant': significant_at_day is not None
        }

    def run_simulation_batch(
        self,
        test_function: callable,
        peek_every: int = 1,
        include_effect: bool = False,
        n_simulations: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run multiple simulations and collect results.

        Args:
            test_function: Statistical test to use
            peek_every: Peeking frequency
            include_effect: A/B vs A/A test
            n_simulations: Override default simulation count

        Returns:
            DataFrame with simulation results
        """
        n = n_simulations or self.config.n_simulations
        results = []

        for sim_id in range(n):
            # Reset RNG with new seed for each simulation
            self.rng = np.random.default_rng(seed=sim_id)

            result = self.run_single_experiment(
                test_function,
                peek_every,
                include_effect
            )
            result['simulation_id'] = sim_id
            results.append(result)

        return pd.DataFrame(results)


def calculate_false_positive_rate(results_df: pd.DataFrame) -> float:
    """Calculate the false positive rate from simulation results."""
    return results_df['ever_significant'].mean()


def calculate_stopping_distribution(results_df: pd.DataFrame) -> pd.Series:
    """Calculate distribution of when tests stopped (if they did)."""
    stopped = results_df[results_df['significant_at_day'].notna()]
    return stopped['significant_at_day'].value_counts().sort_index()
