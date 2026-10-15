"""Approved Phase 2B CPU regression contracts. Missing interfaces are failures.

No model, optimizer, calibration routine, judge client, or dataset is initialized.
"""

import ast
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/v10_boundary_cases.json").read_text())[
    "cases"
]


def canonical_ast_hash(node):
    """Stable across Python AST dump formatting and empty type_params additions."""

    def encode(value):
        if isinstance(value, ast.AST):
            return {
                "node": type(value).__name__,
                "fields": {
                    name: encode(field)
                    for name, field in ast.iter_fields(value)
                    if not (name == "type_params" and not field)
                },
            }
        if isinstance(value, list):
            return [encode(item) for item in value]
        return value

    payload = json.dumps(encode(node), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def pending_symbol(test, module_name, symbol):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        test.fail(f"Required Phase 2B module missing: {module_name}")
    if not hasattr(module, symbol):
        test.fail(f"Required Phase 2B interface missing: {module_name}.{symbol}")
    return getattr(module, symbol)


def pure_function(path, name):
    tree = ast.parse((ROOT / path).read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    scope = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])),
            path,
            "exec",
        ),
        scope,
    )
    return scope[name]


class ApprovedDesignGuards(unittest.TestCase):
    def test_exact_approved_prompt_and_hashes(self):
        proposal = json.loads(
            (ROOT / "docs/references/phase2a/aaai_prompt_proposal.json").read_text()
        )
        self.assertEqual(list(proposal["datasets"]), ["umwp"])
        common = proposal["shared_reasoning_and_abstention_instruction"]
        self.assertNotIn("Keep any required model-specific closing tags", common)
        self.assertIn("Let's think step by step", common)
        self.assertIn("Do not introduce any additional assumptions.", common)
        self.assertIn(
            "REASON: <your explanation here>, then output FINAL: I don't know.", common
        )
        self.assertTrue(common.endswith("output exactly one FINAL: line."))
        prompt = proposal["datasets"]["umwp"]["exact_task_prompt_template"]
        self.assertEqual(
            prompt, "The entire math problem is spoken in the audio.\n\n" + common
        )
        for text, digest in [
            (common, proposal["shared_instruction_sha256"]),
            (prompt, proposal["datasets"]["umwp"]["template_sha256"]),
        ]:
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), digest)
            self.assertIn(text, (ROOT / "AAAI_PROMPT_AUDIT.md").read_text())
        self.assertTrue(proposal["active_generation_prompts_modified"])
        self.assertFalse(proposal["approval_scope"]["comma_checkpoints"])

    def test_frozen_algorithm_asts_without_execution(self):
        guards = json.loads(
            (ROOT / "tests/fixtures/v10_algorithm_guards.json").read_text()
        )
        for guard in guards["checks"]:
            with self.subTest(guard=guard["selector"]):
                tree = ast.parse((ROOT / guard["path"]).read_text())
                node = (
                    tree
                    if guard["selector"] == "module"
                    else next(
                        n
                        for n in tree.body
                        if isinstance(n, ast.FunctionDef)
                        and n.name == guard["selector"]
                    )
                )
                digest = canonical_ast_hash(node)
                self.assertEqual(digest, guard["ast_sha256"])

    def test_validation_choice_and_exact_lower_layer_tie(self):
        tree = ast.parse((ROOT / "src/gate/select_layer.py").read_text())
        run = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"
        )
        choice = next(
            n.value
            for n in ast.walk(run)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "best" for t in n.targets)
        )
        expression = compile(
            ast.Expression(body=choice), "validation_choice_only", "eval"
        )
        rows = [
            {"layer_number": 7, "mean_auc": 0.8},
            {"layer_number": 3, "mean_auc": 0.8},
            {"layer_number": 2, "mean_auc": 0.7},
        ]
        self.assertEqual(eval(expression, {"results": rows})["layer_number"], 3)

    def test_contract_fixtures_are_synthetic_and_cover_required_transitions(self):
        payload = json.loads(
            (ROOT / "tests/fixtures/v10_boundary_cases.json").read_text()
        )
        self.assertTrue(payload["not_dataset_or_model_outputs"])
        names = {case["name"] for case in CASES}
        self.assertEqual(len(names), len(CASES))
        self.assertTrue(
            {
                "fragmented_reason",
                "same_token_stop_veto",
                "ar_response_direct",
                "ar_think_close",
                "comma_excluded",
                "pending_at_end",
            }
            <= names
        )


class BoundaryConfirmationContracts(unittest.TestCase):
    def tracker(self, pieces, model="af_next"):
        cls = pending_symbol(self, "src.models.probing", "BoundaryConfirmationTracker")
        cursor = [1]

        def decode(token):
            self.assertEqual(
                token,
                cursor[0],
                "Decoder may not request future or duplicate token text",
            )
            cursor[0] += 1
            return pieces[token - 1]

        return cls(decode_token=decode, model_name=model)

    def run_case(self, case):
        tracker = self.tracker(case["pieces"], case["model"])
        events = []
        prefix = ""
        for token, piece in enumerate(case["pieces"], 1):
            prefix += piece
            earlier = copy.deepcopy(events)
            observed = tracker.observe(token)
            self.assertLessEqual(len(observed), 1)
            self.assertEqual(
                events, earlier, "Already emitted metadata must not be mutated"
            )
            for event in observed:
                self.assertEqual(event["confirmation_token_index"], token)
                self.assertEqual(event["readout_kind"], "boundary_confirmation_state")
                for boundary in event["boundaries"]:
                    end = boundary["punctuation_char_end"]
                    separator = boundary["confirmation_char_offset"]
                    self.assertEqual(boundary["post_punctuation_text"], prefix[end:])
                    self.assertEqual(
                        boundary["pre_confirmation_text"], prefix[end:separator]
                    )
                    self.assertEqual(
                        boundary["confirmation_delay_tokens"],
                        token - boundary["punctuation_token_index"],
                    )
                events.append(event)
        self.assertEqual(
            [e["confirmation_token_index"] for e in events],
            case["expected_checkpoints"],
        )
        self.assertEqual(
            [len(e["boundaries"]) for e in events], case["expected_boundary_counts"]
        )
        marker = tracker.stop_event["marker"] if tracker.stop_event else None
        self.assertEqual(marker, case["expected_stop_marker"])

    def test_prefix_extension_cannot_retract_prior_checkpoint(self):
        pieces = ["Done.", "\nREA", "SON: Missing. "]
        tracker = self.tracker(pieces)
        self.assertEqual(tracker.observe(1), ())
        events = tracker.observe(2)
        self.assertEqual([e["confirmation_token_index"] for e in events], [2])
        snapshot = copy.deepcopy(events)
        self.assertEqual(tracker.observe(3), ())
        self.assertEqual(events, snapshot)
        self.assertEqual(tracker.stop_event["token_index"], 3)

    def test_confirmation_owner_offsets_and_extra_text(self):
        tracker = self.tracker(["Done", ".", " Next"])
        tracker.observe(1)
        tracker.observe(2)
        boundary = tracker.observe(3)[0]["boundaries"][0]
        self.assertEqual(boundary["punctuation_token_index"], 2)
        self.assertEqual(
            (boundary["punctuation_char_start"], boundary["punctuation_char_end"]),
            (4, 5),
        )
        self.assertEqual(boundary["confirmation_char_offset"], 5)
        self.assertEqual(boundary["confirmation_delay_tokens"], 1)
        self.assertEqual(boundary["pre_confirmation_text"], "")
        self.assertEqual(boundary["post_punctuation_text"], " Next")

    def test_www_prefix_is_masked_before_a_host_is_observed(self):
        tracker = self.tracker(["www", ".", " Next"])
        self.assertEqual(tracker.observe(1), ())
        self.assertEqual(tracker.observe(2), ())
        self.assertEqual(tracker.observe(3), ())

    def test_native_stop_id_precedes_generic_special_token_masking(self):
        cls = pending_symbol(self, "src.models.probing", "BoundaryConfirmationTracker")

        def decode(token):
            self.assertEqual(
                token,
                1,
                "Native stop ID must be recognized without decoding/masking it",
            )
            return "Reasoning. "

        tracker = cls(
            decode_token=decode,
            model_name="audio-reasoner",
            native_stop_token_ids={99: "<RESPONSE>"},
        )
        self.assertEqual(tracker.observe(1)[0]["confirmation_token_index"], 1)
        self.assertEqual(tracker.observe(99), ())
        self.assertEqual(tracker.stop_event["marker"], "<RESPONSE>")
        self.assertEqual(tracker.stop_event["token_index"], 2)


for _case in CASES:

    def _contract(self, case=_case):
        self.run_case(case)

    setattr(BoundaryConfirmationContracts, "test_" + _case["name"], _contract)


class DecoderAndPromptContracts(unittest.TestCase):
    def test_incremental_utf8_never_emits_replacement_characters(self):
        cls = pending_symbol(
            self, "src.models.token_text", "IncrementalUTF8TextDecoder"
        )
        decoder = cls()
        raw = "값".encode()
        self.assertEqual(decoder.feed_bytes(raw[:1]), "")
        self.assertEqual(decoder.feed_bytes(raw[1:2]), "")
        self.assertEqual(decoder.feed_bytes(raw[2:] + b". "), "값. ")
        with self.assertRaises(UnicodeDecodeError):
            cls().feed_bytes(b"\xff")

    def test_runtime_task_provider_returns_the_approved_bytes(self):
        provider = pending_symbol(
            self, "src.capture.v10_prompt", "get_umwp_task_prompt"
        )
        expected = json.loads(
            (ROOT / "docs/references/phase2a/aaai_prompt_proposal.json").read_text()
        )["datasets"]["umwp"]["exact_task_prompt_template"]
        self.assertEqual(provider(), expected)
        self.assertEqual(provider(), provider())


class EmptyStateAndReplayContracts(unittest.TestCase):
    def test_zero_checkpoint_output_is_preserved_without_intervention(self):
        helper = pending_symbol(
            self, "src.capture.v10_eligibility", "zero_checkpoint_record"
        )
        for condition in ("answerable", "unanswerable"):
            trial = {
                "trial_id": "synthetic",
                "condition": condition,
                "text": "FINAL: I don't know.",
                "token_ids": [11, 21, 99],
                "stop_reason": "eos",
            }
            snapshot = copy.deepcopy(trial)
            row = helper(trial)
            self.assertEqual(trial, snapshot)
            for field in trial:
                self.assertEqual(row[field], trial[field])
            self.assertEqual(row["gate_output_text"], trial["text"])
            self.assertEqual(row["token_ids"], trial["token_ids"])
            self.assertEqual(row["trajectory_score"], 0.0)
            self.assertEqual(row["eligible_checkpoint_count"], 0)
            self.assertEqual(
                row["monitoring_status"], "no_eligible_reasoning_checkpoint"
            )
            self.assertFalse(row["triggered"])
            self.assertFalse(row["intervention_run"])
            self.assertIsNone(row["brake_checkpoint"])
            self.assertIsNone(row["intervention_result"])

    def test_exact_replay_prefix_requires_the_confirmation_state(self):
        helper = pending_symbol(
            self, "src.capture.v10_eligibility", "resolve_confirmed_prefix"
        )
        ids = [21, 21, 8, 99]
        self.assertEqual(helper(ids, 3, {1, 3}), [21, 21, 8])
        for index, available in [(0, {1}), (5, {1}), (4, {1, 3})]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                helper(ids, index, available)

    def test_actual_empty_score_paths_short_circuit_before_gate_calls(self):
        pending_symbol(self, "src.models.probing", "BoundaryConfirmationTracker")

        class EmptyProbabilities:
            def reshape(self, *args):
                return self

            def numel(self):
                return 0

        class EmptyStates:
            shape = (0, 32, 8)

            def __getitem__(self, key):
                raise AssertionError("Empty-state path must not prepare a gate input")

        def forbidden_gate(*args):
            raise AssertionError("No gate call is allowed for zero checkpoints")

        select = pure_function("src/gate/select_layer.py", "trajectory_score")
        calibrate_score = pure_function(
            "src/gate/calibrate_tau5.py", "trajectory_score"
        )
        self.assertEqual(select(EmptyProbabilities()), 0.0)
        self.assertEqual(
            calibrate_score(EmptyStates(), 1, forbidden_gate, None, None, None), 0.0
        )
        for name in ("run_val_intervention", "run_test_intervention"):
            crossing = pure_function(
                f"src/intervention/{name}.py", "first_crossing_details"
            )
            self.assertIsNone(
                crossing(EmptyStates(), 1, forbidden_gate, None, None, 0.5)
            )


class PairAndThresholdContracts(unittest.TestCase):
    def test_whole_pair_filter_and_condition_specific_exclusion_counts(self):
        helper = pending_symbol(
            self, "src.gate.v10_eligibility", "filter_pairs_for_state_fitting"
        )
        pairs = [
            {
                "pair_id": str(i),
                "answerable": {"eligible_checkpoint_count": a},
                "unanswerable": {"eligible_checkpoint_count": u},
            }
            for i, (a, u) in enumerate([(2, 3), (0, 3), (2, 0), (0, 0)])
        ]
        snapshot = copy.deepcopy(pairs)
        for model in ("af_next", "audio-reasoner", "moss"):
            kept, report = helper(pairs, model)
            self.assertEqual(pairs, snapshot)
            self.assertEqual([p["pair_id"] for p in kept], ["0"])
            self.assertEqual(report["model"], model)
            self.assertEqual(report["n_pairs"], 4)
            self.assertEqual(report["excluded_pair_count"], 3)
            self.assertEqual(report["excluded_pair_rate"], 0.75)
            for condition in ("answerable", "unanswerable"):
                stats = report["condition_stats"][condition]
                self.assertEqual(stats["total_trial_count"], 4)
                self.assertEqual(stats["zero_checkpoint_count"], 2)
                self.assertEqual(stats["zero_checkpoint_rate"], 0.5)
                self.assertEqual(stats["state_fitting_exclusion_count"], 3)
                self.assertEqual(stats["state_fitting_exclusion_rate"], 0.75)
                self.assertEqual(stats["partner_only_exclusion_count"], 1)

    def test_negative_count_is_corruption_not_coverage_loss(self):
        helper = pending_symbol(
            self, "src.gate.v10_eligibility", "filter_pairs_for_state_fitting"
        )
        with self.assertRaises(ValueError):
            helper(
                [
                    {
                        "pair_id": "bad",
                        "answerable": {"eligible_checkpoint_count": -1},
                        "unanswerable": {"eligible_checkpoint_count": 2},
                    }
                ],
                "moss",
            )

    def test_missing_count_cannot_be_silently_treated_as_zero(self):
        helper = pending_symbol(
            self, "src.gate.v10_eligibility", "filter_pairs_for_state_fitting"
        )
        with self.assertRaises(ValueError):
            helper(
                [
                    {
                        "pair_id": "bad",
                        "answerable": {},
                        "unanswerable": {"eligible_checkpoint_count": 2},
                    }
                ],
                "af_next",
            )

    def test_no_complete_pair_returns_auditable_total_exclusion(self):
        helper = pending_symbol(
            self, "src.gate.v10_eligibility", "filter_pairs_for_state_fitting"
        )
        pairs = [
            {
                "pair_id": "empty",
                "answerable": {"eligible_checkpoint_count": 0},
                "unanswerable": {"eligible_checkpoint_count": 0},
            }
        ]
        kept, report = helper(pairs, "audio-reasoner")
        self.assertEqual(kept, [])
        self.assertEqual(report["excluded_pair_count"], 1)
        self.assertEqual(report["excluded_pair_rate"], 1.0)

    def test_positive_float64_threshold_without_calibrating_anything(self):
        helper = pending_symbol(
            self, "src.gate.v10_eligibility", "validate_positive_threshold"
        )
        positive = math.nextafter(0.0, math.inf)
        self.assertEqual(helper(positive), positive)
        self.assertEqual(helper(0.5), 0.5)
        for value in (0.0, -0.1, math.nan, math.inf, 2.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                helper(value)


class WrapperAwareResponseContracts(unittest.TestCase):
    def inspect(self, text, model="audio-reasoner"):
        helper = pending_symbol(
            self, "src.eval.response_format", "inspect_response_format"
        )
        return helper(text, model_name=model)

    def test_native_closer_on_final_line_is_not_answer_content(self):
        report = self.inspect(
            "<RESPONSE>\nREASON: The inputs imply 42.\nFINAL: 42</RESPONSE>"
        )
        self.assertEqual(report["final_text"], "42")
        self.assertTrue(report["format_ok"])

    def test_blank_final_cannot_consume_a_closing_tag(self):
        report = self.inspect("FINAL:\n</RESPONSE>")
        self.assertIsNone(report["final_text"])
        self.assertFalse(report["format_ok"])

    def test_direct_abstention_keeps_stance_but_flags_missing_reason(self):
        report = self.inspect("FINAL: I don't know.", model="af_next")
        self.assertEqual(report["final_text"], "I don't know.")
        self.assertFalse(report["format_ok"])
        self.assertFalse(report["reason_present"])

    def test_duplicate_or_nonterminal_final_is_a_format_violation(self):
        for text in ("FINAL: 17\nFINAL: 42", "FINAL: 42\nMore ordinary answer prose."):
            self.assertFalse(self.inspect(text)["format_ok"])

    def test_arbitrary_angle_brackets_in_an_answer_are_preserved(self):
        report = self.inspect("FINAL: <2>", model="af_next")
        self.assertEqual(report["final_text"], "<2>")
        self.assertTrue(report["format_ok"])


if __name__ == "__main__":
    unittest.main()
