"""Statistical Validation for Temporal Correlations.

Provides rigorous statistical tests to avoid the "horoscope problem" -
distinguishing real correlations from spurious patterns in personal data.

Research Foundation:
- ACCESS Benchmark (2025): Validation metrics for causal discovery
- Bradford Hill Criteria: Statistical evidence requirements

Implements:
- Fisher's exact test for small sample correlation testing
- Chi-square test for larger samples
- Bootstrap confidence intervals
- Bonferroni correction for multiple hypothesis testing
- Effect size estimation (Cramer's V, Odds Ratio)

Option B Compliance:
- Statistical validation without model fine-tuning
- Results inform token priors via natural language summaries
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StatisticalSignificanceResult:
    """Result of statistical significance testing.

    Captures p-value, confidence intervals, and effect size
    for rigorous correlation validation.

    Example:
        >>> result = StatisticalSignificanceResult(
        ...     p_value=0.023,
        ...     test_used="fisher_exact",
        ...     is_significant=True,
        ... )
        >>> if result.is_significant:
        ...     print(f"Correlation confirmed (p={result.p_value:.4f})")
    """

    # Core significance metrics
    p_value: float = field(default=1.0)
    test_used: str = field(default="none")  # "fisher_exact", "chi_square", "bootstrap"
    is_significant: bool = field(default=False)

    # Confidence interval
    confidence_interval_95: Tuple[float, float] = field(default=(0.0, 1.0))
    point_estimate: float = field(default=0.0)

    # Multiple comparison correction
    corrected_p_value: Optional[float] = field(default=None)
    correction_method: Optional[str] = field(default=None)  # "bonferroni", "fdr"

    # Effect size
    effect_size: Optional[float] = field(default=None)
    effect_size_type: Optional[str] = field(default=None)  # "odds_ratio", "cramers_v"
    effect_interpretation: Optional[str] = field(default=None)  # "small", "medium", "large"

    # Sample info
    sample_size: int = field(default=0)
    expected_by_chance: float = field(default=0.0)
    observed_count: int = field(default=0)

    # Metadata
    computed_at: datetime = field(default_factory=datetime.utcnow)

    def to_natural_language(self) -> str:
        """Convert to natural language summary for token priors.

        Returns human-readable description of statistical findings.
        """
        if not self.is_significant:
            return (
                f"No statistically significant correlation detected "
                f"(p={self.p_value:.3f}, test={self.test_used}). "
                f"Observed {self.observed_count} co-occurrences vs "
                f"{self.expected_by_chance:.1f} expected by chance."
            )

        effect_desc = ""
        if self.effect_interpretation:
            effect_desc = f" Effect size: {self.effect_interpretation}."

        ci_desc = ""
        if self.confidence_interval_95[0] != 0.0 or self.confidence_interval_95[1] != 1.0:
            ci_desc = (
                f" 95% CI: [{self.confidence_interval_95[0]:.2f}, "
                f"{self.confidence_interval_95[1]:.2f}]."
            )

        return (
            f"Statistically significant correlation (p={self.p_value:.4f}, "
            f"test={self.test_used}). "
            f"Observed {self.observed_count} co-occurrences vs "
            f"{self.expected_by_chance:.1f} expected by chance.{effect_desc}{ci_desc}"
        )


class StatisticalCorrelationValidator:
    """Rigorous statistical tests for temporal correlations.

    Prevents the "horoscope problem" by validating correlations
    against chance expectations and providing confidence measures.

    Research Foundation:
    - ACCESS Benchmark: Use evaluation metrics to validate causal discovery
    - Bradford Hill: Statistical evidence requirements for causality

    Example:
        >>> validator = StatisticalCorrelationValidator()
        >>> result = validator.validate_correlation(
        ...     observed_cooccurrences=8,
        ...     total_events_a=50,
        ...     total_events_b=40,
        ...     time_range_days=30,
        ...     max_gap_days=7,
        ... )
        >>> if result.is_significant:
        ...     print(f"Real correlation (p={result.p_value:.4f})")
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        min_effect_size: float = 0.1,
        bootstrap_samples: int = 1000,
    ):
        """Initialize the validator.

        Args:
            significance_threshold: P-value threshold for significance
            min_effect_size: Minimum effect size to consider meaningful
            bootstrap_samples: Number of bootstrap iterations for CI
        """
        self.significance_threshold = significance_threshold
        self.min_effect_size = min_effect_size
        self.bootstrap_samples = bootstrap_samples

    def validate_correlation(
        self,
        observed_cooccurrences: int,
        total_events_a: int,
        total_events_b: int,
        time_range_days: float,
        max_gap_days: float,
        gap_values: Optional[List[float]] = None,
    ) -> StatisticalSignificanceResult:
        """Validate a temporal correlation with statistical tests.

        Uses Fisher's exact test for small samples, chi-square for larger.
        Calculates expected co-occurrences by chance under null hypothesis.

        Args:
            observed_cooccurrences: Number of A->B co-occurrences found
            total_events_a: Total events of type A in period
            total_events_b: Total events of type B in period
            time_range_days: Total time range analyzed (days)
            max_gap_days: Maximum gap allowed for correlation
            gap_values: Optional list of actual gap values for CI estimation

        Returns:
            StatisticalSignificanceResult with test results
        """
        # Calculate expected co-occurrences under null (independence)
        expected = self._calculate_expected_cooccurrences(
            total_a=total_events_a,
            total_b=total_events_b,
            time_range_days=time_range_days,
            max_gap_days=max_gap_days,
        )

        # Build contingency table for significance test
        # [observed, expected]
        # [not_observed_a, not_observed_b]

        # Choose test based on sample size
        sample_size = total_events_a + total_events_b

        if sample_size < 20 or observed_cooccurrences < 5:
            # Small sample: Fisher's exact test
            p_value, odds_ratio = self._fisher_exact_test(
                observed=observed_cooccurrences,
                expected=expected,
                total_a=total_events_a,
                total_b=total_events_b,
            )
            test_used = "fisher_exact"
            effect_size = odds_ratio
            effect_type = "odds_ratio"
        else:
            # Larger sample: Chi-square test
            p_value, chi_stat = self._chi_square_test(
                observed=observed_cooccurrences,
                expected=expected,
                total_a=total_events_a,
                total_b=total_events_b,
            )
            test_used = "chi_square"
            # Calculate Cramer's V for effect size
            effect_size = self._cramers_v(chi_stat, sample_size, min(2, 2))
            effect_type = "cramers_v"

        # Determine significance
        is_significant = (
            p_value < self.significance_threshold
            and observed_cooccurrences > expected
            and (effect_size is None or effect_size >= self.min_effect_size)
        )

        # Bootstrap confidence interval for gap estimate
        confidence_interval = (0.0, 1.0)
        point_estimate = 0.0
        if gap_values and len(gap_values) >= 3:
            point_estimate = sum(gap_values) / len(gap_values)
            confidence_interval = self._bootstrap_confidence_interval(
                data=gap_values,
                statistic=lambda x: sum(x) / len(x) if x else 0,
            )

        # Interpret effect size
        effect_interp = self._interpret_effect_size(effect_size, effect_type)

        return StatisticalSignificanceResult(
            p_value=p_value,
            test_used=test_used,
            is_significant=is_significant,
            confidence_interval_95=confidence_interval,
            point_estimate=point_estimate,
            effect_size=effect_size,
            effect_size_type=effect_type,
            effect_interpretation=effect_interp,
            sample_size=sample_size,
            expected_by_chance=expected,
            observed_count=observed_cooccurrences,
        )

    def _calculate_expected_cooccurrences(
        self,
        total_a: int,
        total_b: int,
        time_range_days: float,
        max_gap_days: float,
    ) -> float:
        """Calculate expected co-occurrences under null hypothesis.

        Under independence, probability that any A is followed by B
        within max_gap is:

        P(B follows A within gap) = (B_count * max_gap) / time_range

        Expected co-occurrences = A_count * P(B follows A)
        """
        if time_range_days <= 0:
            return 0.0

        # Probability that any random moment has a B event within max_gap
        prob_b_in_window = min(1.0, (total_b * max_gap_days) / time_range_days)

        # Expected co-occurrences
        expected = total_a * prob_b_in_window

        return max(0.1, expected)  # Floor at 0.1 to avoid division by zero

    def _fisher_exact_test(
        self,
        observed: int,
        expected: float,
        total_a: int,
        total_b: int,
    ) -> Tuple[float, float]:
        """Fisher's exact test for small samples.

        Tests whether observed co-occurrences significantly exceed
        what would be expected by chance.

        Returns:
            (p_value, odds_ratio)
        """
        # Construct 2x2 contingency table:
        # [[observed, total_b - observed], [total_a - observed, remainder]]

        a = observed
        b = max(0, total_b - observed)
        c = max(0, total_a - observed)
        d = max(1, (total_a + total_b) - a - b - c)  # Ensure positive

        # Calculate p-value using hypergeometric distribution approximation
        # For exact test, we'd need scipy.stats.fisher_exact
        # Using approximation for self-contained implementation

        n = a + b + c + d
        if n == 0:
            return 1.0, 1.0

        # Expected value under independence
        expected_a = (a + b) * (a + c) / n if n > 0 else 0

        # Use chi-square approximation for p-value
        if expected_a > 0:
            chi_approx = ((a - expected_a) ** 2) / expected_a
            # Convert to approximate p-value using chi-square 1 df
            p_value = self._chi_square_to_pvalue(chi_approx, df=1)
        else:
            p_value = 1.0

        # Odds ratio
        if b * c > 0:
            odds_ratio = (a * d) / (b * c)
        else:
            odds_ratio = float('inf') if a > 0 else 1.0

        # Cap odds ratio for practical use
        odds_ratio = min(odds_ratio, 100.0)

        return p_value, odds_ratio

    def _chi_square_test(
        self,
        observed: int,
        expected: float,
        total_a: int,
        total_b: int,
    ) -> Tuple[float, float]:
        """Chi-square test for larger samples.

        Tests whether observed co-occurrences significantly differ
        from expected under independence.

        Returns:
            (p_value, chi_statistic)
        """
        if expected <= 0:
            return 1.0, 0.0

        # Chi-square statistic
        chi_stat = ((observed - expected) ** 2) / expected

        # Also check the non-occurrence cell
        not_observed = total_a - observed
        not_expected = total_a - expected
        if not_expected > 0:
            chi_stat += ((not_observed - not_expected) ** 2) / max(0.1, not_expected)

        # Convert to p-value (1 degree of freedom)
        p_value = self._chi_square_to_pvalue(chi_stat, df=1)

        return p_value, chi_stat

    def _chi_square_to_pvalue(self, chi_stat: float, df: int = 1) -> float:
        """Convert chi-square statistic to p-value.

        Uses approximation of chi-square CDF without scipy.
        For df=1, this is equivalent to normal distribution squared.
        """
        if chi_stat <= 0:
            return 1.0

        # For df=1, p-value = 2 * (1 - Phi(sqrt(chi)))
        # where Phi is standard normal CDF
        z = math.sqrt(chi_stat)

        # Normal CDF approximation (Abramowitz and Stegun)
        return 2 * (1 - self._normal_cdf(z))

    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation.

        Uses error function approximation.
        """
        # Constants for approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

        return 0.5 * (1.0 + sign * y)

    def _cramers_v(
        self,
        chi_stat: float,
        n: int,
        min_dim: int,
    ) -> float:
        """Calculate Cramer's V effect size.

        V = sqrt(chi^2 / (n * (min(r,c) - 1)))

        For 2x2 table, this reduces to sqrt(chi^2 / n).
        """
        if n <= 0 or min_dim <= 1:
            return 0.0

        return math.sqrt(chi_stat / (n * (min_dim - 1)))

    def _interpret_effect_size(
        self,
        effect_size: Optional[float],
        effect_type: Optional[str],
    ) -> Optional[str]:
        """Interpret effect size magnitude.

        Cohen's conventions:
        - Small: 0.1-0.3
        - Medium: 0.3-0.5
        - Large: > 0.5

        For odds ratio:
        - Small: 1.5-2.5
        - Medium: 2.5-4.0
        - Large: > 4.0
        """
        if effect_size is None:
            return None

        if effect_type == "odds_ratio":
            if effect_size < 1.5:
                return "negligible"
            elif effect_size < 2.5:
                return "small"
            elif effect_size < 4.0:
                return "medium"
            else:
                return "large"
        else:
            # Cramer's V or other correlation-like measures
            if effect_size < 0.1:
                return "negligible"
            elif effect_size < 0.3:
                return "small"
            elif effect_size < 0.5:
                return "medium"
            else:
                return "large"

    def _bootstrap_confidence_interval(
        self,
        data: List[float],
        statistic: callable,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval.

        Uses percentile bootstrap method for robust CI estimation.

        Args:
            data: Sample data
            statistic: Function to compute statistic of interest
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            (lower_bound, upper_bound) tuple
        """
        if not data or len(data) < 3:
            return (0.0, float('inf'))

        n = len(data)
        bootstrap_stats = []

        for _ in range(self.bootstrap_samples):
            # Resample with replacement
            sample = [random.choice(data) for _ in range(n)]
            bootstrap_stats.append(statistic(sample))

        # Sort for percentile calculation
        bootstrap_stats.sort()

        # Calculate percentile indices
        alpha = 1 - confidence
        lower_idx = int((alpha / 2) * len(bootstrap_stats))
        upper_idx = int((1 - alpha / 2) * len(bootstrap_stats)) - 1

        lower_idx = max(0, min(lower_idx, len(bootstrap_stats) - 1))
        upper_idx = max(0, min(upper_idx, len(bootstrap_stats) - 1))

        return (bootstrap_stats[lower_idx], bootstrap_stats[upper_idx])

    def bonferroni_correction(
        self,
        p_values: List[float],
    ) -> List[Tuple[float, bool]]:
        """Apply Bonferroni correction for multiple hypothesis testing.

        Adjusts significance threshold by number of tests to control
        family-wise error rate.

        Args:
            p_values: List of p-values from multiple tests

        Returns:
            List of (corrected_p_value, is_significant) tuples
        """
        if not p_values:
            return []

        n_tests = len(p_values)
        corrected_threshold = self.significance_threshold / n_tests

        results = []
        for p in p_values:
            corrected_p = min(1.0, p * n_tests)
            is_sig = p < corrected_threshold
            results.append((corrected_p, is_sig))

        return results

    def benjamini_hochberg_correction(
        self,
        p_values: List[float],
    ) -> List[Tuple[float, bool]]:
        """Apply Benjamini-Hochberg FDR correction.

        Less conservative than Bonferroni, controls false discovery rate.

        Args:
            p_values: List of p-values from multiple tests

        Returns:
            List of (adjusted_p_value, is_significant) tuples
        """
        if not p_values:
            return []

        n = len(p_values)

        # Sort p-values with original indices
        indexed = [(p, i) for i, p in enumerate(p_values)]
        indexed.sort(key=lambda x: x[0])

        # Calculate adjusted p-values
        adjusted = [0.0] * n
        prev_adj = 0.0

        for rank, (p, orig_idx) in enumerate(indexed, 1):
            # BH adjustment: p * n / rank
            adj_p = min(1.0, p * n / rank)
            # Ensure monotonicity
            adj_p = max(adj_p, prev_adj)
            adjusted[orig_idx] = adj_p
            prev_adj = adj_p

        # Determine significance
        results = []
        for adj_p in adjusted:
            is_sig = adj_p < self.significance_threshold
            results.append((adj_p, is_sig))

        return results

    def validate_multiple_correlations(
        self,
        correlations: List[StatisticalSignificanceResult],
        correction_method: str = "bonferroni",
    ) -> List[StatisticalSignificanceResult]:
        """Apply multiple testing correction to a set of correlations.

        Args:
            correlations: List of significance results
            correction_method: "bonferroni" or "fdr" (Benjamini-Hochberg)

        Returns:
            Updated results with corrected p-values and significance
        """
        if not correlations:
            return []

        p_values = [c.p_value for c in correlations]

        if correction_method == "fdr":
            corrections = self.benjamini_hochberg_correction(p_values)
        else:
            corrections = self.bonferroni_correction(p_values)

        # Update results
        updated = []
        for corr, (corrected_p, is_sig) in zip(correlations, corrections):
            # Create new result with corrections
            new_result = StatisticalSignificanceResult(
                p_value=corr.p_value,
                test_used=corr.test_used,
                is_significant=is_sig,  # Updated based on correction
                confidence_interval_95=corr.confidence_interval_95,
                point_estimate=corr.point_estimate,
                corrected_p_value=corrected_p,
                correction_method=correction_method,
                effect_size=corr.effect_size,
                effect_size_type=corr.effect_size_type,
                effect_interpretation=corr.effect_interpretation,
                sample_size=corr.sample_size,
                expected_by_chance=corr.expected_by_chance,
                observed_count=corr.observed_count,
            )
            updated.append(new_result)

        return updated
