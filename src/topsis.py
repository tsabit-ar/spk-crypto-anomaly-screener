"""TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) Decision Engine.

Ranks cryptocurrency anomaly candidates based on 5 multi-criteria attributes:
- C1: Funding Rate (%) [Cost, weight=0.25]
- C2: Delta OI 4H (%) [Benefit, weight=0.25]
- C3: Bollinger Band Width 1H (%) [Cost, weight=0.20]
- C4: Depth Imbalance Ratio [Benefit, weight=0.15]
- C5: Volume / OI Velocity [Benefit, weight=0.15]
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS: Dict[str, float] = {
    "C1": 0.25,  # Funding Rate (%) -> Cost
    "C2": 0.25,  # Delta OI 4H (%) -> Benefit
    "C3": 0.20,  # BBW 1H (%) -> Cost
    "C4": 0.15,  # Depth Imbalance -> Benefit
    "C5": 0.15,  # Volume/OI Velocity -> Benefit
}

DEFAULT_CRITERIA_TYPES: Dict[str, str] = {
    "C1": "cost",
    "C2": "benefit",
    "C3": "cost",
    "C4": "benefit",
    "C5": "benefit",
}

EPSILON = 1e-9


class TopsisEngine:
    """TOPSIS Multi-Criteria Decision Making (MCDM) Engine."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        criteria_types: Optional[Dict[str, str]] = None,
    ):
        """Initialize TOPSIS engine with weights and criteria definitions.

        Args:
            weights: Dictionary of criteria weights (must sum to ~1.0).
            criteria_types: Dictionary mapping criteria to 'benefit' or 'cost'.
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.criteria_types = criteria_types or DEFAULT_CRITERIA_TYPES.copy()
        self.criteria_keys = list(self.weights.keys())

        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def rank_candidates(self, df_candidates: pd.DataFrame) -> pd.DataFrame:
        """Execute TOPSIS ranking algorithm on candidates DataFrame.

        Args:
            df_candidates: DataFrame containing columns 'symbol' and 'C1'..'C5'.

        Returns:
            pd.DataFrame: Ranked DataFrame sorted by score (Ci) with 'Rank' column.
        """
        if df_candidates.empty:
            return df_candidates

        # Ensure all criteria columns are present
        for col in self.criteria_keys:
            if col not in df_candidates.columns:
                raise ValueError(f"Missing required criteria column: '{col}' in input DataFrame.")

        df = df_candidates.copy().reset_index(drop=True)
        matrix = df[self.criteria_keys].to_numpy(dtype=np.float64)
        m, n = matrix.shape

        if m == 0:
            return df

        # Step 1: Vector Normalization (Euclidean norm per column)
        # r_ij = x_ij / sqrt(sum(x_kj^2))
        col_norms = np.sqrt(np.sum(matrix ** 2, axis=0))
        # Handle zero-norm edge case with epsilon
        col_norms = np.where(col_norms == 0, EPSILON, col_norms)
        norm_matrix = matrix / col_norms

        # Step 2: Weighted Normalized Decision Matrix
        # v_ij = w_j * r_ij
        weight_vec = np.array([self.weights[k] for k in self.criteria_keys], dtype=np.float64)
        weighted_matrix = norm_matrix * weight_vec

        # Step 3: Determine Ideal Positive (A+) and Ideal Negative (A-) Solutions
        ideal_pos = np.zeros(n, dtype=np.float64)
        ideal_neg = np.zeros(n, dtype=np.float64)

        for j, crit in enumerate(self.criteria_keys):
            crit_type = self.criteria_types.get(crit, "benefit").lower()
            col_values = weighted_matrix[:, j]
            if crit_type == "benefit":
                ideal_pos[j] = np.max(col_values)
                ideal_neg[j] = np.min(col_values)
            else:  # cost
                ideal_pos[j] = np.min(col_values)
                ideal_neg[j] = np.max(col_values)

        # Step 4: Calculate Euclidean Distances (D+ and D-)
        # D+_i = sqrt(sum((v_ij - A+_j)^2))
        # D-_i = sqrt(sum((v_ij - A-_j)^2))
        d_pos = np.sqrt(np.sum((weighted_matrix - ideal_pos) ** 2, axis=1))
        d_neg = np.sqrt(np.sum((weighted_matrix - ideal_neg) ** 2, axis=1))

        # Step 5: Calculate Relative Closeness / Preference Score (Ci)
        # Ci = D-_i / (D+_i + D-_i)
        denom = d_pos + d_neg
        denom = np.where(denom == 0, EPSILON, denom)
        ci_scores = d_neg / denom

        # Step 6: Add scores and rank
        df["D_pos"] = d_pos
        df["D_neg"] = d_neg
        df["topsis_score"] = ci_scores

        # Sort descending by topsis_score
        df_ranked = df.sort_values(by="topsis_score", ascending=False).reset_index(drop=True)
        df_ranked["rank"] = df_ranked.index + 1

        return df_ranked
