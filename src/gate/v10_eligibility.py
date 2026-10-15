"""Coverage accounting outside the unchanged gate fitting/calibration kernels."""

import math


def validate_positive_threshold(tau):
    tau = float(tau)
    if not math.isfinite(tau) or not 0 < tau <= math.nextafter(1.0, math.inf):
        raise ValueError(
            "V10 thresholds must be finite, strictly positive float64 values"
        )
    return tau


def filter_pairs_for_state_fitting(pairs, model_name):
    counts = []
    for pair in pairs:
        row = []
        for condition in ("answerable", "unanswerable"):
            value = pair[condition].get("eligible_checkpoint_count")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "Missing/invalid eligible_checkpoint_count is corruption"
                )
            row.append(value)
        counts.append(row)
    kept = [pair for pair, (a, u) in zip(pairs, counts) if a > 0 and u > 0]
    n = len(pairs)
    excluded = n - len(kept)
    stats = {}
    for index, condition in enumerate(("answerable", "unanswerable")):
        zero = sum(row[index] == 0 for row in counts)
        stats[condition] = dict(
            total_trial_count=n,
            zero_checkpoint_count=zero,
            zero_checkpoint_rate=zero / n if n else 0.0,
            state_fitting_exclusion_count=excluded,
            state_fitting_exclusion_rate=excluded / n if n else 0.0,
            partner_only_exclusion_count=excluded - zero,
        )
    return kept, dict(
        model=model_name,
        n_pairs=n,
        excluded_pair_count=excluded,
        excluded_pair_rate=excluded / n if n else 0.0,
        condition_stats=stats,
    )
