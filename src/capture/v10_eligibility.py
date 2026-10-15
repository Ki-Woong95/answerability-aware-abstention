"""Zero coverage is a completed behavioral trial, not a failed capture."""


def zero_checkpoint_record(trial):
    return {
        **trial,
        "gate_output_text": trial["text"],
        "trajectory_score": 0.0,
        "eligible_checkpoint_count": 0,
        "monitoring_status": "no_eligible_reasoning_checkpoint",
        "triggered": False,
        "brake_checkpoint": None,
        "intervention_run": False,
        "intervention_result": None,
    }


def resolve_confirmed_prefix(ids, k, available_positions):
    if not isinstance(k, int) or k <= 0 or k > len(ids) or k not in available_positions:
        raise ValueError(
            "Replay requires an available confirmation state and exact token prefix"
        )
    return list(ids[:k])
