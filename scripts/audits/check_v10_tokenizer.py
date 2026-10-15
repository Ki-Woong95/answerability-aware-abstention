"""Offline tokenizer-only audit, no model initialization."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from transformers import AutoTokenizer
from src.models.token_text import TokenTextDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--path", required=True)
parser.add_argument("--slow", action="store_true")
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(
    args.path, local_files_only=True, use_fast=not args.slow
)
fixtures = [
    "Done. Next; reconsider! Why? ",
    "값. 다음 답변입니다. ",
    "3.14; Dr. Smith; e.g. examples. ",
    "<THINK><REASONING>Enough. </REASONING><SUMMARY>Continue. </THINK><RESPONSE>\nFINAL: 42",
    "REASON: Missing information.\nFINAL: I don't know.",
]
for text in fixtures:
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoder = TokenTextDecoder(tokenizer)
    pieces = [decoder(i) for i in ids]
    assert "\ufffd" not in "".join(pieces)
    assert "".join(pieces) == tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
report = dict(
    model=args.model,
    tokenizer_path=args.path,
    slow=args.slow,
    tokenizer_fingerprint=decoder.fingerprint,
    fixtures_passed=len(fixtures),
    native_stop_token_ids=decoder.native_stop_token_ids,
    byte_decode_verified=True,
    model_inference=False,
)
Path(f"docs/provenance/phase2b/{args.model}_tokenizer.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
