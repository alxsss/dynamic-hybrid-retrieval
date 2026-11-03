"""Deterministic Unicode-aware tokenization for lexical retrieval."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Final, Literal

TOKEN_PATTERN: Final = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Versioned settings that determine index compatibility."""

    version: int = 1
    unicode_normalization: Literal["NFC", "NFKC"] = "NFKC"
    casefold: bool = True

    def as_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)


class Tokenizer:
    """Small deterministic tokenizer shared by indexing and querying."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        if self.config.version != 1:
            raise ValueError(f"unsupported tokenizer version {self.config.version}")
        if self.config.unicode_normalization not in {"NFC", "NFKC"}:
            raise ValueError("unicode_normalization must be NFC or NFKC")

    def normalize(self, text: str) -> str:
        normalized = unicodedata.normalize(self.config.unicode_normalization, text)
        return normalized.casefold() if self.config.casefold else normalized

    def tokenize(self, text: str) -> tuple[str, ...]:
        normalized = self.normalize(text)
        return tuple(match.group(0) for match in TOKEN_PATTERN.finditer(normalized))
