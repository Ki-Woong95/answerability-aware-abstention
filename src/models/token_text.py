"""Append-only UTF-8 decoding for audited byte-level BPE tokenizers."""

import codecs


class IncrementalUTF8TextDecoder:
    def __init__(self, errors="strict"):
        if errors not in ("strict", "replace"):
            raise ValueError("Only strict or native replacement decoding is supported")
        self.errors = errors
        self.last_error = None
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors=errors)

    def feed_bytes(self, raw, final=False):
        self.last_error = None
        if self.errors == "replace":
            probe = codecs.getincrementaldecoder("utf-8")(errors="strict")
            probe.setstate(self.decoder.getstate())
            try:
                probe.decode(raw, final=final)
            except UnicodeDecodeError as exc:
                self.last_error = dict(
                    raw_bytes_hex=raw.hex(),
                    buffered_bytes_hex=self.decoder.getstate()[0].hex(),
                    error_start=exc.start,
                    error_end=exc.end,
                    reason=exc.reason,
                )
        return self.decoder.decode(raw, final=final)


class TokenTextDecoder:
    """Read raw token bytes; delay incomplete UTF-8 until it is decided causally.

    Supports the byte-level BPE used by the three Qwen-family backends.
    Unknown tokenizer families fail closed rather than changing prefix offsets.
    Optional native replacement mode records malformed bytes; strict is default.
    """

    def __init__(self, tokenizer, errors="strict"):
        self.tokenizer = tokenizer
        self.utf8 = IncrementalUTF8TextDecoder(errors)
        self.observed_token_ids = []
        self.text = ""
        self.replacement_events = []
        values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        chars = values[:]
        n = 0
        for value in range(256):
            if value not in values:
                values.append(value)
                chars.append(256 + n)
                n += 1
        self.inverse = dict(zip(map(chr, chars), values))
        self.special = set(tokenizer.all_special_ids)
        self.native_stop_token_ids = {
            int(i): tokenizer.convert_ids_to_tokens(int(i))
            for i in self.special
            if tokenizer.convert_ids_to_tokens(int(i)).upper()
            in ("</THINK>", "<RESPONSE>")
        }
        import hashlib

        self.fingerprint = hashlib.sha256(
            str(sorted(tokenizer.get_vocab().items())).encode()
        ).hexdigest()
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            import json

            config = json.loads(backend.to_str())
            if config.get("decoder", {}).get("type") != "ByteLevel":
                raise ValueError("V10 requires an audited ByteLevel tokenizer decoder")
        elif not hasattr(tokenizer, "byte_decoder"):
            raise ValueError("Tokenizer raw-byte decoding is not audited")

    def __call__(self, token_id):
        self.observed_token_ids.append(int(token_id))
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        if int(token_id) in self.special:
            # A observed special marker ends the byte run. Strict mode fails
            # closed; native mode commits any malformed pending bytes here.
            if self.utf8.decoder.getstate()[0]:
                if self.utf8.errors == "strict":
                    raise ValueError("Special token interrupts incomplete UTF-8")
                piece = self.utf8.feed_bytes(b"", final=True) + token
            else:
                self.utf8.last_error = None
                piece = token
        else:
            piece = self.utf8.feed_bytes(bytes(self.inverse[c] for c in token))
        if self.utf8.last_error:
            self.replacement_events.append(
                dict(
                    self.utf8.last_error,
                    token_index=len(self.observed_token_ids),
                    token_id=int(token_id),
                    emitted_text=piece,
                )
            )
        self.text += piece
        return piece
