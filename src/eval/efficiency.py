"""Definitions for output-length metrics; these are not timing measurements."""

GENERATED_OUTPUT_TOKEN_METRIC = {
    "unit": "generated_output_tokens",
    "post_intervention_generated_tokens": "brake_checkpoint + num_continuation_tokens",
    "generated_token_reduction_vs_vanilla": "vanilla_num_generated_tokens - post_intervention_generated_tokens",
    "scope": "Saved generated prefix plus emitted continuation, or unchanged Vanilla output when not triggered.",
    "not_measured": [
        "GPU compute",
        "latency",
        "audio/prompt/prefix prefill",
        "gate scoring",
        "injected instruction tokens",
    ],
}
