"""Reference vectors as kernel-checked Lean examples.

A vectors file has one line per vector, `OP FIELD operand... result`, in hexadecimal. Each
vector inside the contract of its operation becomes an example that asks the kernel to evaluate
the transcription on the operands and compare with the recorded result; the others are counted
and left out, since the proofs say nothing about them.
"""
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Operation:
    """How a vector's operation is checked: the Lean function evaluated, and the contract
    `(modulus, operands) -> bool` its operands must satisfy."""

    function: str
    contract: Callable


@dataclass(frozen=True)
class Field:
    """A field a vector names: its constants' Lean name and its modulus."""

    lean: str
    modulus: int


def examples(lines, fields, operations):
    """The body of the vectors module: one example per vector inside its contract, then the
    count of vectors and of those left out."""
    out, n, skipped = [], 0, {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        op, key, *values = parts
        field = fields[key]
        if op not in operations:
            raise ValueError(line)
        *operands, result = values
        if not operations[op].contract(field.modulus, [int(v, 16) for v in operands]):
            skipped[op] = skipped.get(op, 0) + 1
            continue
        # One operand per line keeps every line within the repository's width.
        out.append(f"example :\n    {operations[op].function}\n")
        out += [f"      (Limbs.ofNat 0x{v})\n" for v in operands]
        out.append(f"      {field.lean}.modulus {field.lean}.inv =\n"
                   f"    (Limbs.ofNat 0x{result}) := by\n  decide +kernel\n\n")
        n += 1
    omitted = ", ".join(f"{count} {op}" for op, count in sorted(skipped.items())) or "none"
    out.append(f"\n-- {n} vectors; omitted as outside the proved contracts: {omitted}.\n\n")
    return "".join(out)
