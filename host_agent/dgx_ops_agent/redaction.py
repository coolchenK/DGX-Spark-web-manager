"""Streaming credential redaction for untrusted process output."""

from __future__ import annotations

import codecs
import re
from collections.abc import Iterable

REDACTED = "[REDACTED]"
_NORMAL_CARRY = 96
_MAX_DIRECT_KEY_LENGTH = 256
_JSON_WHITESPACE = frozenset(" \t\r\n")
_TOKEN_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._~+/=-"
)
_UNQUOTED_TERMINATORS = frozenset(" \t\r\n,;&")

_AUTHORIZATION = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(authorization\s*:\s*(?:bearer|basic)\s+)(?=\S)"
)
_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"((?:[A-Za-z][A-Za-z0-9]*[_-])*"
    r"(?:secret[_-]access[_-]key|private[_-]key|api[_-]?key|access[_-]?token|"
    r"auth[_-]?token|token|password|passwd|secret)"
    r"['\"]?\s*[:=]\s*(?P<quote>['\"]?))(?=\S)"
)
_ASSIGNMENT_KEY = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"((?:[A-Za-z][A-Za-z0-9]*[_-])*"
    r"(?:secret[_-]access[_-]key|private[_-]key|api[_-]?key|access[_-]?token|"
    r"auth[_-]?token|token|password|passwd|secret)['\"])(?![A-Za-z0-9])"
)
_DIRECT_TOKEN = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:github_pat_|gh[oprsu]_|hf_|dgx_|sk-|xox[baprs]-)"
    r"(?=[A-Za-z0-9])"
)


class StreamingRedactor:
    """Redact credentials without exposing markers split across byte chunks."""

    def __init__(self, *, secret_values: Iterable[str] = ()) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""
        self._mode = "normal"
        self._quote = ""
        self._quoted_escape = False
        self._direct_text: str | None = ""
        self._direct_preservable = False
        self._direct_key_quote = ""
        self._key_whitespace_count = 0
        self._finished = False
        secrets = tuple(value for value in secret_values if value)
        self._secret_pattern = (
            re.compile(
                "|".join(
                    re.escape(value)
                    for value in sorted(secrets, key=len, reverse=True)
                )
            )
            if secrets
            else None
        )
        self._normal_carry = max((_NORMAL_CARRY, *(len(value) for value in secrets)))

    def feed(self, chunk: bytes) -> bytes:
        """Consume one bytes chunk and return only output safe to publish."""

        if self._finished:
            raise RuntimeError("redactor is already finished")
        if not isinstance(chunk, bytes):
            raise TypeError("chunk must be bytes")
        self._buffer += self._decoder.decode(chunk, final=False)
        return self._scan(final=False).encode("utf-8")

    def finish(self) -> bytes:
        """Flush the decoder and redact a credential that ends at EOF."""

        if self._finished:
            return b""
        self._finished = True
        self._buffer += self._decoder.decode(b"", final=True)
        return self._scan(final=True).encode("utf-8")

    def _scan(self, *, final: bool) -> str:
        output: list[str] = []
        while self._buffer or (final and self._mode != "normal"):
            if self._mode != "normal":
                if not self._consume_secret(output, final=final):
                    break
                continue

            candidate = self._next_marker()
            if candidate is None:
                if final:
                    output.append(self._buffer)
                    self._buffer = ""
                elif len(self._buffer) > self._normal_carry:
                    stable = len(self._buffer) - self._normal_carry
                    output.append(self._buffer[:stable])
                    self._buffer = self._buffer[stable:]
                break

            kind, match = candidate
            output.append(self._buffer[: match.start()])
            marker = match.group(0)
            preceding = self._buffer[match.start() - 1] if match.start() else ""
            self._buffer = self._buffer[match.end() :]
            if kind in {"direct", "literal"}:
                self._mode = "direct"
                if len(marker) > _MAX_DIRECT_KEY_LENGTH:
                    output.append(REDACTED)
                    self._direct_text = None
                else:
                    self._direct_text = marker
                self._direct_preservable = kind == "direct"
                self._direct_key_quote = (
                    preceding if preceding in {'"', "'"} else ""
                )
            elif kind == "assignment_key":
                output.append(marker)
                self._mode = "key_colon"
            else:
                output.append(marker)
                if kind == "assignment" and match.groupdict().get("quote"):
                    output.append(REDACTED)
                    self._mode = "quoted"
                    self._quote = match.group("quote")
                    self._quoted_escape = False
                elif kind == "assignment":
                    self._mode = "assignment_value"
                else:
                    output.append(REDACTED)
                    self._mode = kind
        return "".join(output)

    def _next_marker(self) -> tuple[str, re.Match[str]] | None:
        candidates: list[tuple[str, re.Match[str]]] = []
        for kind, pattern in (
            ("authorization", _AUTHORIZATION),
            ("assignment_key", _ASSIGNMENT_KEY),
            ("assignment", _ASSIGNMENT),
            ("direct", _DIRECT_TOKEN),
        ):
            match = pattern.search(self._buffer)
            if match is not None:
                candidates.append((kind, match))
        if self._secret_pattern is not None:
            match = self._secret_pattern.search(self._buffer)
            if match is not None:
                for _kind, candidate in candidates:
                    if candidate.start() <= match.start() < candidate.end():
                        return "literal", match
                candidates.append(("literal", match))
        if not candidates:
            return None
        priority = {"literal": 0, "direct": 1, "assignment_key": 2}
        return min(
            candidates,
            key=lambda item: (item[1].start(), priority.get(item[0], 3)),
        )

    def _consume_secret(self, output: list[str], *, final: bool) -> bool:
        if self._mode == "direct":
            return self._consume_direct(output, final=final)

        if self._mode == "key_colon":
            return self._consume_key_colon(output, final=final)

        if self._mode == "direct_key":
            return self._consume_direct_key(output, final=final)

        if self._mode == "assignment_value":
            return self._consume_assignment_value(output, final=final)

        if self._mode == "quoted":
            delimiter, trailing_escape = self._find_unescaped_quote(
                self._buffer,
                self._quote,
                self._quoted_escape,
            )
            if delimiter is None:
                self._buffer = ""
                self._quoted_escape = False if final else trailing_escape
                return False
            output.append(self._quote)
            self._buffer = self._buffer[delimiter + 1 :]
            self._mode = "normal"
            self._quote = ""
            self._quoted_escape = False
            return True

        index = next(
            (
                index
                for index, character in enumerate(self._buffer)
                if character in _UNQUOTED_TERMINATORS
            ),
            None,
        )
        if index is None:
            self._buffer = ""
            if final:
                self._mode = "normal"
            return False

        output.append(self._buffer[index])
        self._buffer = self._buffer[index + 1 :]
        self._mode = "normal"
        return True

    def _consume_direct(self, output: list[str], *, final: bool) -> bool:
        index = next(
            (
                index
                for index, character in enumerate(self._buffer)
                if character not in _TOKEN_CHARACTERS
            ),
            None,
        )
        token_part = self._buffer if index is None else self._buffer[:index]
        self._extend_direct_text(token_part, output)
        if index is None:
            self._buffer = ""
            if final:
                self._emit_direct_text(output)
                self._reset_direct()
            return False

        terminator = self._buffer[index]
        self._buffer = self._buffer[index + 1 :]
        if terminator == self._direct_key_quote:
            self._mode = "direct_key"
            self._quote = terminator
            self._key_whitespace_count = 0
            return True

        self._emit_direct_text(output)
        output.append(terminator)
        self._reset_direct()
        return True

    def _consume_key_colon(self, output: list[str], *, final: bool) -> bool:
        whitespace_length = 0
        while (
            whitespace_length < len(self._buffer)
            and self._buffer[whitespace_length] in _JSON_WHITESPACE
        ):
            whitespace_length += 1
        output.append(self._buffer[:whitespace_length])
        self._buffer = self._buffer[whitespace_length:]

        if not self._buffer:
            if final:
                self._mode = "normal"
            return False

        if self._buffer[0] == ":":
            output.append(":")
            self._buffer = self._buffer[1:]
            self._mode = "assignment_value"
            return True

        self._mode = "normal"
        return True

    def _consume_direct_key(self, output: list[str], *, final: bool) -> bool:
        whitespace_length = 0
        while (
            whitespace_length < len(self._buffer)
            and self._buffer[whitespace_length] in _JSON_WHITESPACE
        ):
            whitespace_length += 1
        self._key_whitespace_count += whitespace_length
        self._buffer = self._buffer[whitespace_length:]

        if not self._buffer:
            if final:
                self._emit_direct_text(output, preserve=False)
                output.extend((self._quote, " " * self._key_whitespace_count))
                self._reset_direct()
            return False

        if self._buffer[0] == ":":
            self._emit_direct_text(output, preserve=True)
            output.extend((self._quote, " " * self._key_whitespace_count, ":"))
            self._buffer = self._buffer[1:]
            self._reset_direct(mode="assignment_value")
            return True

        self._emit_direct_text(output, preserve=False)
        output.extend((self._quote, " " * self._key_whitespace_count))
        self._reset_direct()
        return True

    def _consume_assignment_value(
        self,
        output: list[str],
        *,
        final: bool,
    ) -> bool:
        whitespace_length = 0
        while (
            whitespace_length < len(self._buffer)
            and self._buffer[whitespace_length] in _JSON_WHITESPACE
        ):
            whitespace_length += 1
        output.append(self._buffer[:whitespace_length])
        self._buffer = self._buffer[whitespace_length:]
        if not self._buffer:
            if final:
                self._mode = "normal"
            return False

        if self._buffer[0] in {'"', "'"}:
            self._quote = self._buffer[0]
            self._buffer = self._buffer[1:]
            output.extend((self._quote, REDACTED))
            self._quoted_escape = False
            self._mode = "quoted"
        else:
            output.append(REDACTED)
            self._mode = "assignment"
        return True

    def _extend_direct_text(self, value: str, output: list[str]) -> None:
        if self._direct_text is None:
            return
        if len(self._direct_text) + len(value) <= _MAX_DIRECT_KEY_LENGTH:
            self._direct_text += value
            return
        output.append(REDACTED)
        self._direct_text = None

    def _emit_direct_text(self, output: list[str], *, preserve: bool = False) -> None:
        if self._direct_text is None:
            return
        output.append(
            self._direct_text
            if preserve and self._direct_preservable
            else REDACTED
        )

    def _reset_direct(self, *, mode: str = "normal") -> None:
        self._mode = mode
        self._direct_text = ""
        self._direct_preservable = False
        self._direct_key_quote = ""
        self._quote = ""
        self._key_whitespace_count = 0

    @staticmethod
    def _find_unescaped_quote(
        value: str,
        quote: str,
        escaped: bool,
    ) -> tuple[int | None, bool]:
        for index, character in enumerate(value):
            if character == "\\":
                escaped = not escaped
                continue
            if character == quote and not escaped:
                return index, False
            escaped = False
        return None, escaped


def redact_text(value: str) -> str:
    """Redact a complete string using the same rules as streamed output."""

    redactor = StreamingRedactor()
    return (redactor.feed(value.encode("utf-8", errors="replace")) + redactor.finish()).decode(
        "utf-8"
    )


__all__ = ["REDACTED", "StreamingRedactor", "redact_text"]
