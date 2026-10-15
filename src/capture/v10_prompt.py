"""AAAI-style instruction; common initial input for Vanilla and Gate."""

PROMPT_VERSION = "v10-aaai-umwp-reason-before-final-v4"
TASK_PROMPT = """The entire math problem is spoken in the audio.

Let's think step by step and output the final answer on a separate last answer line in the format FINAL: <answer>.
Please solve the problem strictly based on the information provided. Do not introduce any additional assumptions.
If you believe the problem lacks sufficient information or is unsolvable, explain why on a separate line in the format REASON: <your explanation here>, then output FINAL: I don't know.
Place REASON: before FINAL: and output exactly one FINAL: line."""


def get_umwp_task_prompt():
    return TASK_PROMPT


# Keep reasoning/abstention bytes identical after reassessment.
SHARED_REASONING_INSTRUCTION = TASK_PROMPT.split("\n\n", 1)[1]
INTERVENTION_PROMPT_VERSION = "v10-umwp-conditional-reassessment-v1"
INTERVENTION_PROMPT = (
    "Reassess whether the audio contains sufficient information to answer the question."
    "\n\n" + SHARED_REASONING_INSTRUCTION
)


def intervention_prompt_provenance():
    import hashlib

    return {
        "intervention_prompt_version": INTERVENTION_PROMPT_VERSION,
        "intervention_prompt_sha256": hashlib.sha256(
            INTERVENTION_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
