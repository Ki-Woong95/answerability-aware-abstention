"""Shared native generation termination for AR capture and continuation."""

RESPONSE_END = "</RESPONSE>"
STOP_POLICY = "ar-native-response-end-v1"


def request_stop_sequences():
    return [RESPONSE_END]


def stop_policy_provenance(model_name):
    return (
        {"generation_stop_policy": STOP_POLICY}
        if model_name == "audio-reasoner"
        else {}
    )


def classify_stop(token_ids, text, budget, eos_ids, tokenizer):
    # Stream output may omit a matched stop string. The emitted IDs are retained.
    decode = getattr(tokenizer, "decode", None)
    raw = (
        decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decode
        else text
    )
    if RESPONSE_END in raw:
        return "stop_sequence", RESPONSE_END
    eos_ids = eos_ids if isinstance(eos_ids, (list, tuple, set)) else [eos_ids]
    if token_ids and token_ids[-1] in eos_ids:
        return "eos", None
    if len(token_ids) >= budget:
        return "max_new_tokens", None
    return "unknown", None
