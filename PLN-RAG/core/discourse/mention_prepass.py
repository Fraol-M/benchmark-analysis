from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable


PRONOUNS = {
    "he",
    "her",
    "him",
    "his",
    "it",
    "its",
    "she",
    "their",
    "them",
    "they",
}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those"}
STOPWORDS = {
    "and",
    "are",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "near",
    "not",
    "of",
    "on",
    "or",
    "over",
    "than",
    "to",
    "was",
    "were",
    "who",
    "with",
}


@dataclass
class Mention:
    id: str
    text: str
    canonical: str
    kind: str
    sentence_index: int
    start: int
    end: int
    candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MentionPrepassResult:
    mentions: list[Mention] = field(default_factory=list)

    @property
    def ambiguous_pronouns(self) -> list[Mention]:
        return [
            mention
            for mention in self.mentions
            if mention.kind == "pronoun" and len(mention.candidates) > 1
        ]

    def to_dict(self) -> dict:
        return {
            "mentions": [mention.to_dict() for mention in self.mentions],
            "ambiguous_pronouns": [
                mention.to_dict() for mention in self.ambiguous_pronouns
            ],
        }

    def prompt_hint(self, *, max_mentions: int = 16) -> str:
        if not self.mentions:
            return ""

        lines = [
            "",
            "",
            "Mention prepass hints:",
            "- Mentions below are deterministic anchors, not facts.",
            "- If a pronoun has multiple candidates, do not replace it with one candidate.",
            "- For ambiguous pronouns, preserve the pronoun mention id as the argument "
            "(for example it_m4) instead of guessing camera or phone.",
        ]
        for mention in self.mentions[:max_mentions]:
            candidate_text = ""
            if mention.candidates:
                candidate_text = (
                    " candidates="
                    + ",".join(self._candidate_labels(mention.candidates))
                )
            lines.append(
                f"- {mention.id}: {mention.kind} text={mention.text!r} "
                f"canonical={mention.canonical!r}{candidate_text}"
            )
        return "\n".join(lines)

    def _candidate_labels(self, ids: Iterable[str]) -> list[str]:
        by_id = {mention.id: mention for mention in self.mentions}
        return [
            f"{candidate_id}:{by_id[candidate_id].canonical}"
            for candidate_id in ids
            if candidate_id in by_id
        ]


class MentionPrepass:
    """Small deterministic mention scan used to preserve ambiguity for LangExtract."""

    def build(self, text: str) -> MentionPrepassResult:
        mentions: list[Mention] = []
        prior_nominals: list[Mention] = []
        sentence_spans = list(self._sentence_spans(text))
        mention_number = 1

        for sentence_index, start, end in sentence_spans:
            sentence = text[start:end]
            tokens = list(self._tokens(sentence, offset=start))
            token_mentions: list[Mention] = []

            for idx, (token, token_start, token_end) in enumerate(tokens):
                lower = token.lower()
                if lower in PRONOUNS:
                    candidates = [
                        mention.id
                        for mention in prior_nominals[-6:]
                        if self._pronoun_compatible(lower, mention)
                    ]
                    mention = Mention(
                        id=f"{lower}_m{mention_number}",
                        text=token,
                        canonical=lower,
                        kind="pronoun",
                        sentence_index=sentence_index,
                        start=token_start,
                        end=token_end,
                        candidates=candidates,
                    )
                    mention_number += 1
                    mentions.append(mention)
                    token_mentions.append(mention)
                    continue

                if token[:1].isupper() and not self._is_sentence_initial(
                    token_start,
                    start,
                    text,
                ):
                    mention = self._new_nominal(
                        mention_number,
                        token,
                        "name",
                        sentence_index,
                        token_start,
                        token_end,
                    )
                    mention_number += 1
                    mentions.append(mention)
                    token_mentions.append(mention)
                    continue

                if lower in DETERMINERS and idx + 1 < len(tokens):
                    noun, noun_start, noun_end = tokens[idx + 1]
                    noun_lower = noun.lower()
                    if self._is_nominal_token(noun_lower):
                        mention = self._new_nominal(
                            mention_number,
                            noun,
                            "definite" if lower == "the" else "nominal",
                            sentence_index,
                            noun_start,
                            noun_end,
                        )
                        mention_number += 1
                        mentions.append(mention)
                        token_mentions.append(mention)

            prior_nominals.extend(
                mention for mention in token_mentions if mention.kind != "pronoun"
            )

        return MentionPrepassResult(mentions=mentions)

    def _new_nominal(
        self,
        number: int,
        text: str,
        kind: str,
        sentence_index: int,
        start: int,
        end: int,
    ) -> Mention:
        canonical = self._canonical(text)
        return Mention(
            id=f"{canonical}_m{number}",
            text=text,
            canonical=canonical,
            kind=kind,
            sentence_index=sentence_index,
            start=start,
            end=end,
        )

    def _sentence_spans(self, text: str) -> Iterable[tuple[int, int, int]]:
        start = 0
        sentence_index = 0
        for match in re.finditer(r"[.!?](?:\s+|$)", text):
            end = match.end()
            if text[start:end].strip():
                yield sentence_index, start, end
                sentence_index += 1
            start = end
        if text[start:].strip():
            yield sentence_index, start, len(text)

    def _tokens(self, text: str, *, offset: int) -> Iterable[tuple[str, int, int]]:
        for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9_-]*\b", text):
            yield match.group(0), offset + match.start(), offset + match.end()

    def _canonical(self, value: str) -> str:
        value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
        value = value.replace("-", "_")
        value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_").lower()
        return self._singularize(value)

    def _singularize(self, value: str) -> str:
        if len(value) > 3 and value.endswith("s") and not value.endswith(("ss", "us")):
            return value[:-1]
        return value

    def _is_nominal_token(self, token: str) -> bool:
        return (
            len(token) > 2
            and token not in STOPWORDS
            and token not in PRONOUNS
            and token not in DETERMINERS
        )

    def _is_sentence_initial(self, token_start: int, sentence_start: int, text: str) -> bool:
        return text[sentence_start:token_start].strip() == ""

    def _pronoun_compatible(self, pronoun: str, mention: Mention) -> bool:
        if pronoun in {"it", "its"}:
            return mention.kind in {"definite", "nominal"}
        return True
