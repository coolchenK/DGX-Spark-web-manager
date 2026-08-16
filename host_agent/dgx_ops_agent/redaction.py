"""Streaming credential redaction for untrusted process output."""

from __future__ import annotations

import codecs
import re

REDACTED = "[REDACTED]"
_NORMAL_CARRY = 96
_TOKEN_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._~+/=-"
)
_UNQUOTED_TERMINATORS = frozenset(" \t\r\n,;&")

_AUTHORIZATION = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(authorization\s*:\s*(?:bearer|basic)\s+)(?=\S)"
)
_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9_])"
    r"((?:api[_-]?key|access[_-]?token|auth[_-]?token|token|password|passwd|secret)"
    r"\s*[:=]\s*(?P<quote>['\"]?))(?=\S)"
)
_DIRECT_TOKEN = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:github_pat_|gh[oprsu]_|hf_|dgx_|sk-|xox[baprs]-)"
    r"(?=[A-Za-z0-9])"
)


class StreamingRedactor:
    """Redact credentials without exposing markers split across byte chunks."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""
        self._mode = "normal"
        self._quote = ""
        self._finished = False

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
        while self._buffer:
            if self._mode != "normal":
                if not self._consume_secret(output, final=final):
                    break
                continue

            candidate = self._next_marker()
            if candidate is None:
                if final:
                    output.append(self._buffer)
                    self._buffer = ""
                elif len(self._buffer) > _NORMAL_CARRY:
                    stable = len(self._buffer) - _NORMAL_CARRY
                    output.append(self._buffer[:stable])
                    self._buffer = self._buffer[stable:]
                break

            kind, match = candidate
            output.append(self._buffer[: match.start()])
            marker = match.group(0)
            self._buffer = self._buffer[match.end() :]
            if kind == "direct":
                output.append(REDACTED)
                self._mode = "direct"
            else:
                output.append(marker)
                output.append(REDACTED)
                if kind == "assignment" and match.groupdict().get("quote"):
                    self._mode = "quoted"
                    self._quote = match.group("quote")
                else:
                    self._mode = kind
        return "".join(output)

    def _next_marker(self) -> tuple[str, re.Match[str]] | None:
        candidates: list[tuple[str, re.Match[str]]] = []
        for kind, pattern in (
            ("authorization", _AUTHORIZATION),
            ("assignment", _ASSIGNMENT),
            ("direct", _DIRECT_TOKEN),
        ):
            match = pattern.search(self._buffer)
            if match is not None:
                candidates.append((kind, match))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1].start())

    def _consume_secret(self, output: list[str], *, final: bool) -> bool:
        if self._mode == "quoted":
            delimiter = self._find_unescaped_quote(self._buffer, self._quote)
            if delimiter is None:
                self._buffer = "" if final else self._buffer[-1:]
                return False
            output.append(self._quote)
            self._buffer = self._buffer[delimiter + 1 :]
            self._mode = "normal"
            self._quote = ""
            return True

        if self._mode == "direct":
            index = next(
                (
                    index
                    for index, character in enumerate(self._buffer)
                    if character not in _TOKEN_CHARACTERS
                ),
                None,
            )
        else:
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

    @staticmethod
    def _find_unescaped_quote(value: str, quote: str) -> int | None:
        escaped = False
        for index, character in enumerate(value):
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                return index
        return None


def redact_text(value: str) -> str:
    """Redact a complete string using the same rules as streamed output."""

    redactor = StreamingRedactor()
    return (redactor.feed(value.encode("utf-8", errors="replace")) + redactor.finish()).decode(
        "utf-8"
    )


__all__ = ["REDACTED", "StreamingRedactor", "redact_text"]
