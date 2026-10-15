import re
from collections.abc import Callable, Iterable


SENTENCE_FINAL_PUNCTUATION = (".", "?", "!")


class SentenceBoundaryTracker:

    def __init__(
        self,
        decode_token: Callable[[int], str],
        punctuation: tuple[str, ...] = SENTENCE_FINAL_PUNCTUATION,
    ):
        self.decode_token = decode_token
        self.punctuation = punctuation
        self.current_line = ""

    def observe(self, token_id: int) -> bool:
        piece = self.decode_token(int(token_id))

        if not isinstance(piece, str):
            raise TypeError(
                "decode_token must return a string, " f"got {type(piece)!r}"
            )

        context = self.current_line + piece
        stripped_piece = piece.rstrip()

        is_boundary = False

        if stripped_piece.endswith(self.punctuation):
            line_tail = context.rstrip().rsplit("\n", 1)[-1].strip()

            is_numbered_list_marker = re.fullmatch(r"\d+\.", line_tail) is not None

            is_boundary = not is_numbered_list_marker

        if "\n" in context:
            self.current_line = context.rsplit("\n", 1)[-1]
        else:
            self.current_line = context

        return is_boundary


def sentence_boundary_positions(
    token_ids: list[int],
    decode_token: Callable[[int], str],
    punctuation: tuple[str, ...] = SENTENCE_FINAL_PUNCTUATION,
) -> tuple[int, ...]:

    tracker = SentenceBoundaryTracker(
        decode_token=decode_token,
        punctuation=punctuation,
    )

    positions = []

    for position, token_id in enumerate(
        token_ids,
        start=1,
    ):
        if tracker.observe(token_id):
            positions.append(position)

    return tuple(positions)


def select_probe_positions(
    available_positions: Iterable[int],
    fixed_positions: Iterable[int],
    sentence_positions: Iterable[int],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:

    available = set(int(x) for x in available_positions)

    fixed = tuple(int(x) for x in fixed_positions if int(x) in available)

    sentence = tuple(int(x) for x in sentence_positions if int(x) in available)

    all_positions = tuple(sorted(set(fixed) | set(sentence)))

    return (
        all_positions,
        fixed,
        sentence,
    )


BOUNDARY_VERSION = "v10-confirmation-v1"
ABBREVIATIONS = frozenset(
    "Dr. Mr. Mrs. Ms. Prof. Sr. Jr. St. e.g. i.e. etc. vs. No. Fig. approx. Inc. Ltd. a.m. p.m. kg. g. km. cm. mm. sec. min.".split()
)


class BoundaryConfirmationTracker:
    """Causal character scan with token-level stop veto and delayed confirmation.

    observe returns at most one event: its state belongs to the CURRENT consumed
    generated token, not the earlier punctuation token. No end-of-stream flush.
    """

    def __init__(self, decode_token, model_name, native_stop_token_ids=None):
        self.decode_token = decode_token
        self.model_name = model_name
        self.native_stop_token_ids = native_stop_token_ids or {}
        self.text = ""
        self.position = 0
        self.owners = []
        self.pending = None
        self.stop_event = None
        self.events = []
        self.code_delimiter = None
        self.code_run = ""
        self.in_tag = False
        self.exclusion_counts = {}

    def exclude(self, reason):
        self.exclusion_counts[reason] = self.exclusion_counts.get(reason, 0) + 1
        self.pending = None

    def observe(self, token_id):
        self.position += 1
        if self.stop_event is not None:
            return ()
        if token_id in self.native_stop_token_ids:
            self.stop_event = dict(
                marker=self.native_stop_token_ids[token_id],
                token_index=self.position,
                char_offset=len(self.text),
                source="native_token_id",
                same_token_checkpoint_veto=True,
            )
            self.exclusion_counts["monitoring_stop_token_veto"] = 1
            if self.pending:
                self.exclude("pending_at_stop")
            return ()
        piece = self.decode_token(int(token_id))
        if not isinstance(piece, str):
            raise TypeError("decode_token must return committed text")
        start = len(self.text)
        self.text += piece
        self.owners.extend([self.position] * len(piece))
        stops = list(
            re.finditer(
                r"(?:^|[\r\n])[ \t]*(REASON|FINAL)[ \t]*:", self.text, re.I | re.A
            )
        )
        if self.model_name == "audio-reasoner":
            stops += list(re.finditer(r"</THINK>|<RESPONSE>", self.text, re.I | re.A))
        if stops:
            stop = min(stops, key=lambda m: m.start())
            marker = (
                stop.group(1).upper() + ":" if stop.lastindex else stop.group().upper()
            )
            self.stop_event = dict(
                marker=marker,
                token_index=self.position,
                char_offset=stop.start(),
                source="observed_text",
                same_token_checkpoint_veto=True,
            )
            self.exclusion_counts["monitoring_stop_token_veto"] = 1
            if self.pending:
                self.exclude("pending_at_stop")
            return ()
        confirmed = []
        for i in range(start, len(self.text)):
            c = self.text[i]
            before = self.text[:i]
            # Explicit native tags are not linguistic text.
            if c == "<":
                self.in_tag = True
                self.exclude("tag")
            if self.in_tag:
                if c == ">":
                    self.in_tag = False
                continue
            # Backticks and physical-line-start tilde fences. Runs close only
            # when the same delimiter length is observed (possibly split tokens).
            if c in "`~" and (
                c == "`"
                or self.code_delimiter
                or not before.rsplit("\n", 1)[-1].strip("~")
            ):
                self.exclude("code")
                if self.code_run and self.code_run[0] != c:
                    self.code_run = ""
                self.code_run += c
                continue
            if self.code_run:
                run = self.code_run
                self.code_run = ""
                if self.code_delimiter == run:
                    self.code_delimiter = None
                elif self.code_delimiter is None and (run[0] == "`" or len(run) >= 3):
                    self.code_delimiter = run
            if self.code_delimiter:
                continue
            if self.pending:
                pending = self.pending
                if (
                    c in "?!"
                    and self.text[
                        pending["punctuation_char_start"] : pending[
                            "punctuation_char_end"
                        ]
                    ].strip("?!")
                    == ""
                ):
                    pending["punctuation_char_end"] = i + 1
                    pending["punctuation_token_index"] = self.owners[i]
                    continue
                if c in "\"'”’)]}":
                    continue
                if c.isspace():
                    end = pending["punctuation_char_end"]
                    confirmed.append(
                        {
                            **pending,
                            "confirmation_char_offset": i,
                            "confirmation_delay_tokens": self.position
                            - pending["punctuation_token_index"],
                            "pre_confirmation_text": self.text[end:i],
                            "post_punctuation_text": self.text[end:],
                        }
                    )
                    self.pending = None
                    continue
                self.exclude("lexical_continuation")
            if c not in ".?!;":
                continue
            word = re.split(r"\s", before)[-1].lstrip("\"'“‘([{")
            word = re.sub(r"<[^>]*>", "", word)
            if (word + c).startswith(("http://", "https://", "www.")) or "@" in word:
                self.exclude("url_email")
                continue
            if c == "." and i > 0 and self.text[i - 1] == ".":
                self.exclude("ellipsis")
                continue
            if c == ".":
                dotted = word + "."
                line = before.rsplit("\n", 1)[-1].strip() + "."
                if dotted in ABBREVIATIONS or re.fullmatch(r"(?:[A-Za-z]\.)+", dotted):
                    self.exclude("abbreviation")
                    continue
                if re.fullmatch(r"(?:[0-9]+|[A-Za-z]|[IVXLCDMivxlcdm]+)\.", line):
                    self.exclude("list_marker")
                    continue
            if c == ";" and (
                re.fullmatch(
                    r"[+-]?(?:[0-9]+(?:[.,][0-9]+)*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", word
                )
                or re.search(r"&(?:[A-Za-z][A-Za-z0-9]*|#[0-9]+|#x[0-9A-Fa-f]+)$", word)
            ):
                self.exclude("numeric_or_entity_semicolon")
                continue
            self.pending = dict(
                punctuation_char_start=i,
                punctuation_char_end=i + 1,
                punctuation_token_index=self.owners[i],
                punctuation_start_token_index=self.owners[i],
            )
        if not confirmed:
            return ()
        event = dict(
            confirmation_token_index=self.position,
            readout_kind="boundary_confirmation_state",
            boundaries=confirmed,
        )
        self.events.append(event)
        return (event,)
