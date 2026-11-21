"""Mutation parsing, canonicalization, and sequence construction for SrIRED."""
from __future__ import annotations

import re
from collections.abc import Iterable


CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
_MUTATION = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def parse_mutation(token: str) -> tuple[str, int, str]:
    """Parse one one-indexed substitution token."""
    match = _MUTATION.fullmatch(str(token).strip())
    if match is None:
        raise ValueError(f"invalid canonical substitution: {token!r}")
    source, position, target = match.groups()
    return source, int(position), target


def parse_set(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Return a validated, position-sorted substitution set."""
    if value is None or str(value).strip() in {"", "WT", "nan"}:
        return ()
    tokens = (
        [part for part in re.split(r"[-:]", value) if part]
        if isinstance(value, str)
        else list(value)
    )
    parsed = [parse_mutation(token) for token in tokens]
    positions = [item[1] for item in parsed]
    if len(positions) != len(set(positions)):
        raise ValueError(f"multiple substitutions at one position: {value!r}")
    return tuple(f"{source}{position}{target}" for source, position, target in sorted(parsed, key=lambda item: item[1]))


def canonical(value: str | Iterable[str] | None) -> str:
    """Serialize a substitution set using hyphens and ascending positions."""
    tokens = parse_set(value)
    return "WT" if not tokens else "-".join(tokens)


def apply(wild_type: str, value: str | Iterable[str] | None) -> str:
    """Apply validated substitutions to the wild-type sequence."""
    sequence = list(str(wild_type))
    for token in parse_set(value):
        source, position, target = parse_mutation(token)
        index = position - 1
        if index < 0 or index >= len(sequence):
            raise IndexError(f"mutation outside sequence: {token}")
        if sequence[index] != source:
            raise ValueError(f"wild-type mismatch for {token}: found {sequence[index]}")
        sequence[index] = target
    return "".join(sequence)


def add(value: str | Iterable[str] | None, mutation: str) -> tuple[str, ...] | None:
    """Add a substitution, returning None for duplicate or conflicting positions."""
    current = parse_set(value)
    _, position, _ = parse_mutation(mutation)
    if any(parse_mutation(token)[1] == position for token in current):
        return None
    return parse_set((*current, mutation))


def all_singles(wild_type: str) -> list[str]:
    """Enumerate every canonical non-wild-type single substitution."""
    return [
        f"{source}{position}{target}"
        for position, source in enumerate(str(wild_type), start=1)
        for target in CANONICAL_AA
        if target != source
    ]

