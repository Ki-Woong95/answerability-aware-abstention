# v9_final judge runner

import json
import time
from argparse import ArgumentParser
from pathlib import Path

from openai import OpenAI

from src.gate.datasets import get_dataset_config, validate_artifact_dataset
from src.provenance import fingerprint


DEFAULT_JUDGE_MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "low"
RUBRIC_VERSION = "v2"


SYSTEM_PROMPT = """You are an evaluator for an answerability benchmark.

Your task is to classify the FINAL BEHAVIORAL STANCE of a model response.

Do not evaluate whether the reasoning process is elegant, efficient, or
well-written. Judge what the response ultimately concludes.

Use the terminal conclusion as the highest-priority evidence. Markers such as
FINAL:, SUMMARY:, RESPONSE:, or a clear concluding sentence near the end
should be treated as the final stance.

==================================================
ANSWERABLE ITEMS
==================================================

Allowed labels:

- correct:
  The response ultimately selects the correct answer.

- incorrect:
  The response ultimately selects a specific answer, but that answer is not
  the correct answer.

- abstain:
  The response ultimately declines to select an answer because it says the
  information is insufficient, ambiguous, or does not permit a unique answer.

- nonanswer:
  The response never reaches an interpretable final stance. This includes
  responses that end abruptly, are truncated before reaching a conclusion,
  are malformed, or are irrelevant to the question.

Decision rules for ANSWERABLE items:

1. If the terminal conclusion selects a specific answer, classify it as
   correct or incorrect according to the gold answer.

2. Hedging still counts as selecting an answer. Examples such as
   "probably Tom", "I would choose Tom", or "Tom seems most likely"
   are specific answers.

3. If the terminal conclusion explicitly says that the answer cannot be
   determined and does not select a candidate, classify it as abstain.

4. If no clear terminal conclusion is reached, classify it as nonanswer.

5. Listing multiple unresolved candidates is not correct merely because the
   gold answer appears among them.

==================================================
UNANSWERABLE ITEMS
==================================================

Allowed labels:

- abstain:
  The response ultimately recognizes that no unique answer can be determined
  and does not select a specific candidate.

- specific_answer:
  The response ultimately selects or asserts a specific answer, even though
  the item is unanswerable.

- nonanswer:
  The response never reaches an interpretable final stance. This includes
  responses that end abruptly, are truncated before reaching a conclusion,
  are malformed, or are irrelevant to the question.

Decision rules for UNANSWERABLE items:

1. FIRST check the terminal conclusion.

2. If the terminal FINAL, SUMMARY, RESPONSE, or clear concluding statement
   selects or asserts any specific candidate, classify as specific_answer.

   This remains specific_answer even if earlier reasoning says the information
   is ambiguous or insufficient.

3. If the terminal conclusion explicitly says that no unique answer can be
   determined and does not ultimately select a candidate, classify as abstain.

4. If there is no clear terminal stance because the response ends abruptly,
   is truncated, malformed, or never reaches a conclusion, classify as
   nonanswer.

5. Do NOT classify a response as abstain if its final summary or conclusion
   ultimately selects a candidate.

==================================================
GENERAL RULES
==================================================

- Judge semantic meaning, not exact wording.
- Ignore XML-like reasoning tags and formatting artifacts.
- Earlier reasoning is subordinate to the terminal conclusion.
- Merely mentioning "I don't know" earlier does not make the response an
  abstention if it later selects an answer.
- Merely mentioning a candidate earlier does not make the response a
  specific answer if the terminal conclusion says the answer cannot be
  determined.
- If GENERATION HIT TOKEN LIMIT is YES, do not automatically assign nonanswer.
  First check whether a clear terminal conclusion was already produced.
  If the response was cut off before reaching a clear terminal stance,
  classify it as nonanswer.

Return ONLY one JSON object in this format:

{
  "label": "...",
  "extracted_answer": "..." or null,
  "note": "brief justification"
}

For extracted_answer:

- If label is correct, incorrect, or specific_answer,
  extracted_answer MUST be a non-empty string containing the specific
  answer or candidate selected in the model's final conclusion.

- If label is abstain or nonanswer,
  extracted_answer MUST be null.

Never return extracted_answer=null for correct, incorrect, or
specific_answer.

Keep note to one short sentence.
"""


def make_key(row):
    seed = row.get("seed")
    seed_part = seed if seed is not None else "none"
    dataset = row.get("dataset", "umwp")
    return f"{dataset}|{row['model']}|{seed_part}|{row['trial_id']}"


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def build_prompt(row):
    condition = row["condition"]
    response_text = row["response_text"]
    hit_token_limit = "YES" if row.get("hit_token_limit", False) else "NO"
    input_prompt = row.get("input_prompt_text")
    prompt_block = ""
    if isinstance(input_prompt, str) and input_prompt.strip():
        prompt_block = f"""
ORIGINAL TASK PROMPT / QUESTION:
---BEGIN TASK---
{input_prompt}
---END TASK---

"""

    if condition == "answerable":
        gold = row.get("answerable_answer")
        acceptable = row.get("possible_answers") or _as_list(gold)

        return f"""ITEM TYPE: ANSWERABLE

Gold answer:
{gold}

Acceptable answer representation(s):
{json.dumps(acceptable, ensure_ascii=False)}
{prompt_block}
If the gold answer is an option index such as 1 or 2, use the original task
prompt/question above to map a semantically stated candidate answer to that
numbered option. Do not require the model to literally output the option number
when its terminal conclusion clearly selects the corresponding option.

GENERATION HIT TOKEN LIMIT:
{hit_token_limit}

MODEL RESPONSE:
---BEGIN RESPONSE---
{response_text}
---END RESPONSE---

Classify the final behavioral stance."""

    if condition == "unanswerable":
        return f"""ITEM TYPE: UNANSWERABLE

This item does not have a uniquely determinable answer from the provided
information.

Do not infer, prefer, or assume any candidate answer.
{prompt_block}
GENERATION HIT TOKEN LIMIT:
{hit_token_limit}

MODEL RESPONSE:
---BEGIN RESPONSE---
{response_text}
---END RESPONSE---

Classify the final behavioral stance."""

    raise ValueError(f"Unknown condition: {condition}")


def parse_json(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end < start:
        raise ValueError(f"No JSON object found: {text[:300]}")

    return json.loads(text[start : end + 1])


def validate_result(row, result):
    condition = row["condition"]
    label = result.get("label")

    if condition == "answerable":
        allowed = {"correct", "incorrect", "abstain", "nonanswer"}
    elif condition == "unanswerable":
        allowed = {"abstain", "specific_answer", "nonanswer"}
    else:
        raise ValueError(f"Unknown condition: {condition}")

    if label not in allowed:
        raise ValueError(f"Invalid label {label!r} for condition {condition}")

    extracted = result.get("extracted_answer")

    if label in {"abstain", "nonanswer"} and extracted is not None:
        raise ValueError(f"{label} must have extracted_answer=null")

    if label in {"correct", "incorrect", "specific_answer"}:
        if extracted is None or not str(extracted).strip():
            raise ValueError(f"{label} must include a non-empty extracted_answer")

    return {
        "label": label,
        "extracted_answer": extracted,
        "note": str(result.get("note", "")).strip(),
    }


def judge_one(client, row, judge_model, retries=3):
    base_prompt = build_prompt(row)
    last_error = None
    budgets = (500, 1000, 2000)

    for attempt in range(retries):
        prompt = base_prompt

        if attempt > 0:
            prompt += f"""

Your previous evaluation failed output validation for this reason:

{last_error}

Return the evaluation again, correcting that formatting/schema error.
Do not change the classification merely because validation failed.
Return ONLY the required JSON object.
"""

        try:
            response = client.responses.create(
                model=judge_model,
                reasoning={"effort": REASONING_EFFORT},
                instructions=SYSTEM_PROMPT,
                input=prompt,
                max_output_tokens=budgets[min(attempt, len(budgets) - 1)],
            )
            parsed = parse_json(response.output_text)
            result = validate_result(row, parsed)
            return result, response.id
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)

    raise RuntimeError(f"Judge failed after {retries} attempts: {last_error}")


def judge_input_fingerprint(row, judge_model):
    return fingerprint(
        {
            "prompt": build_prompt(row),
            "instructions": SYSTEM_PROMPT,
            "judge_model": judge_model,
            "reasoning_effort": REASONING_EFFORT,
            "rubric_version": RUBRIC_VERSION,
        }
    )


def load_completed(path, input_rows=None, judge_model=DEFAULT_JUDGE_MODEL):
    if not path.exists():
        return {}

    completed = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            key = make_key(row)
            if key in completed:
                raise RuntimeError(f"{path}: duplicate judged row {key}")
            completed[key] = row

    if input_rows is not None:
        inputs = {make_key(row): row for row in input_rows}
        if len(inputs) != len(input_rows):
            raise RuntimeError("Duplicate judge input key")
        for key, row in completed.items():
            if key not in inputs or row.get(
                "input_fingerprint"
            ) != judge_input_fingerprint(inputs[key], judge_model):
                raise RuntimeError(
                    f"{path}: saved judgment lacks matching input/rubric/model provenance for {key}; "
                    "use --output with a fresh file"
                )

    return completed


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="umwp",
        choices=("umwp", "worldsense"),
    )
    parser.add_argument(
        "--kind",
        required=True,
        choices=["vanilla", "intervention"],
    )
    parser.add_argument(
        "--split",
        default="val",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Optional fresh judgment output file"
    )

    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    if args.split == "test" and not args.allow_test:
        raise RuntimeError("Refusing to touch test split without --allow-test")

    dataset_config = get_dataset_config(args.dataset)
    input_path = dataset_config.judge_root / f"{args.split}_{args.kind}.jsonl"

    output_path = (
        dataset_config.judge_root
        / f"{args.split}_{args.kind}_judged_{RUBRIC_VERSION}.jsonl"
    )
    if args.output is not None:
        output_path = args.output

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    input_rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in input_rows:
        validate_artifact_dataset(
            row, dataset_config, input_path, model_name=row.get("model")
        )
    if len({make_key(row) for row in input_rows}) != len(input_rows):
        raise RuntimeError("Duplicate judge input key")
    completed = load_completed(output_path, input_rows, args.judge_model)
    client = OpenAI()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("a", encoding="utf-8") as fout,
    ):

        for line in fin:
            if not line.strip():
                continue

            row = json.loads(line)
            validate_artifact_dataset(
                row, dataset_config, input_path, model_name=row.get("model")
            )
            key = make_key(row)

            if key in completed:
                continue

            result, response_id = judge_one(
                client=client,
                row=row,
                judge_model=args.judge_model,
            )

            from src.data.frozen_umwp import v10_artifact_provenance

            output = {
                **v10_artifact_provenance(dataset_config.name, row["model"]),
                "dataset": dataset_config.name,
                "trial_id": row["trial_id"],
                "model": row["model"],
                "condition": row["condition"],
            }

            if "seed" in row:
                output["seed"] = row["seed"]

            for key in (
                "pair_id",
                "family",
                "question_type",
                "n_entities",
                "source",
                "category",
            ):
                if key in row:
                    output[key] = row.get(key)

            output.update(
                {
                    "label": result["label"],
                    "extracted_answer": result["extracted_answer"],
                    "note": result["note"],
                    "hit_token_limit": bool(row.get("hit_token_limit", False)),
                    "judge_model": args.judge_model,
                    "reasoning_effort": REASONING_EFFORT,
                    "rubric_version": RUBRIC_VERSION,
                    "judge_response_id": response_id,
                    "input_fingerprint": judge_input_fingerprint(row, args.judge_model),
                }
            )

            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            fout.flush()
            completed[make_key(row)] = output

            processed += 1

            print(
                f"{processed:04d} "
                f"{row['model']} "
                f"{row['trial_id']} "
                f"-> {result['label']}"
            )

            if args.limit is not None and processed >= args.limit:
                break

    print(f"\nProcessed this run: {processed}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
