"""The intermediate representation: straight-line programs in A-normal form.

A program is a sequence of nodes, each one operation of the Lean semantics
(`PastaAArch64Asm.Semantics`) applied to registers and literals, and a result tuple. The
Lean text of a node is one or more `let` bindings (`Let`), so a program is a term in A-normal
form: every intermediate value is named, and a register that an instruction writes again is
rebound, shadowing the old binding. That is also the form of the generated Lean.

The operations are those of the Lean semantics, not those of an instruction set: a lifter
(`aarch64.py`) maps each instruction to them. Two nodes write a (result, carry) pair, `addc`
and `subc`; their Lean form binds the pair to `s` and then projects it.

Liveness is the backward dataflow analysis on the `let` bindings. On straight-line code one
backward pass reaches the fixed point, so no iteration is needed.

References: C. Flanagan, A. Sabry, B. Duba, M. Felleisen, *The Essence of Compiling with
Continuations*, PLDI 1993 (A-normal form); A. Appel, *Modern Compiler Implementation*,
chapter 10 (liveness).
"""
from dataclasses import dataclass
from enum import Enum

# The register holding the (result, carry) pair of the last `addc` or `subc`.
PAIR = "s"


def is_literal(operand):
    return operand.isdigit()


@dataclass(frozen=True)
class Record:
    """A structured argument of a program: its Lean name and type, its fields, and the
    hypothesis `name.Bounded` that the correctness theorem provides."""

    name: str
    type: str
    fields: tuple
    bound: str

    def projection(self, field_name):
        """The projection out of the `Bounded` conjunction that bounds `field_name`."""
        i = self.fields.index(field_name)
        return ".".join(["2"] * i + (["1"] if i < len(self.fields) - 1 else []))


@dataclass(frozen=True)
class Let:
    """One `let name := expr` of a program's Lean text. `comment` is its trailing comment.
    `dead` says what a binding that nothing reads becomes: `"comment"` (an argument the block
    binds and never reads), `"drop"` (a carry flag nobody reads), or `"error"`."""

    name: str
    expr: str
    comment: str
    reads: frozenset
    dead: str = "error"


@dataclass(frozen=True, kw_only=True)
class Node:
    """One operation. `comment` is the source instruction (or `argument`, or the round it
    stands for); `pc` is the index of that instruction, `None` for nodes with no instruction."""

    comment: str
    pc: int | None = None

    def lets(self):
        raise NotImplementedError

    def _let(self, name, expr, *reads, **kw):
        return Let(name, expr, kw.pop("comment", self.comment),
                   frozenset(r for r in reads if not is_literal(r)), **kw)


# -- arguments ----------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class Load(Node):
    """A field of a record argument: `dest := record.field`."""

    dest: str
    record: Record
    field: str

    def lets(self):
        return [self._let(self.dest, f"{self.record.name}.{self.field}", dead="comment")]


@dataclass(frozen=True, kw_only=True)
class Scalar(Node):
    """A scalar argument `name`, below `2^64` by the hypothesis `bound`."""

    dest: str
    name: str
    bound: str

    def lets(self):
        return [self._let(self.dest, self.name)]


# -- the semantics' operations ------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class Mov(Node):
    dest: str
    src: str

    def lets(self):
        return [self._let(self.dest, self.src, self.src)]


@dataclass(frozen=True, kw_only=True)
class MulLo(Node):
    """`mulLo a b`, the low word of the product."""

    dest: str
    a: str
    b: str

    def lets(self):
        return [self._let(self.dest, f"mulLo {self.a} {self.b}", self.a, self.b)]


@dataclass(frozen=True, kw_only=True)
class MulHi(Node):
    """`umulh a b`, the high word of the product."""

    dest: str
    a: str
    b: str

    def lets(self):
        return [self._let(self.dest, f"umulh {self.a} {self.b}", self.a, self.b)]


@dataclass(frozen=True, kw_only=True)
class Shl(Node):
    dest: str
    a: str
    k: int

    def lets(self):
        return [self._let(self.dest, f"lsl {self.a} {self.k}", self.a)]


@dataclass(frozen=True, kw_only=True)
class Shr(Node):
    dest: str
    a: str
    k: int

    def lets(self):
        return [self._let(self.dest, f"lsr {self.a} {self.k}", self.a)]


@dataclass(frozen=True, kw_only=True)
class _CarryOp(Node):
    """An addition or subtraction with carry-in `cin`, writing `dest` and the flag `carry`."""

    dest: str
    carry: str
    a: str
    b: str
    cin: str
    function = ""  # the Lean function, `addc` or `subc`

    def lets(self):
        return [self._let(PAIR, f"{self.function} {self.a} {self.b} {self.cin}",
                          self.a, self.b, self.cin),
                self._let(self.dest, f"{PAIR}.1", PAIR, comment=f"  `-> {self.dest}"),
                self._let(self.carry, f"{PAIR}.2", PAIR, comment="  `-> carry", dead="drop")]


@dataclass(frozen=True, kw_only=True)
class AddCarry(_CarryOp):
    """`addc a b cin`: `dest` and `carry` are the sum's low word and carry-out."""

    function = "addc"


@dataclass(frozen=True, kw_only=True)
class SubBorrow(_CarryOp):
    """`subc a b cin`: `dest` and `carry` are the difference's low word and no-borrow flag."""

    function = "subc"


@dataclass(frozen=True, kw_only=True)
class AddTrunc(Node):
    """`(addc a b cin).1`: the sum's low word, the carry-out discarded."""

    dest: str
    a: str
    b: str
    cin: str

    def lets(self):
        return [self._let(self.dest, f"(addc {self.a} {self.b} {self.cin}).1",
                          self.a, self.b, self.cin)]


@dataclass(frozen=True, kw_only=True)
class SubFlag(Node):
    """`(subc a b cin).2`: the no-borrow flag of a difference that is discarded."""

    carry: str
    a: str
    b: str
    cin: str

    def lets(self):
        return [self._let(self.carry, f"(subc {self.a} {self.b} {self.cin}).2",
                          self.a, self.b, self.cin, dead="drop")]


class FlagState(Enum):
    """The state of a flag that selects the first operand of a `Select`."""

    CLEAR = "cselLo"  # the Lean function selecting `x` when the flag is `0`
    SET = "cselCs"  # the Lean function selecting `x` when the flag is `1`


@dataclass(frozen=True, kw_only=True)
class Select(Node):
    """`x` when `flag` is in the state `x_when`, else `y`; in the skeleton's form,
    `if flag = 0 then if_zero else if_nonzero`."""

    dest: str
    flag: str
    x: str
    y: str
    x_when: FlagState

    @property
    def if_zero(self):
        return self.x if self.x_when is FlagState.CLEAR else self.y

    @property
    def if_nonzero(self):
        return self.y if self.x_when is FlagState.CLEAR else self.x

    def lets(self):
        return [self._let(self.dest, f"{self.x_when.value} {self.flag} {self.x} {self.y}",
                          self.flag, self.x, self.y)]


# -- calls ---------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class Call(Node):
    """`dest := function fixed... scalars... ⟨state...⟩`: a call of another program, whose
    fixed arguments are the caller's own arguments and whose other arguments are registers."""

    dest: str
    function: str
    fixed: tuple
    scalars: tuple
    state: tuple

    def expr(self, rename=lambda r: r):
        return (f"{self.function} {' '.join(self.fixed)} "
                f"{' '.join(map(rename, self.scalars))} ⟨{', '.join(map(rename, self.state))}⟩")

    def lets(self):
        return [self._let(self.dest, self.expr(), *self.scalars, *self.state)]


@dataclass(frozen=True, kw_only=True)
class Project(Node):
    """A field of a call's result: `dest := record.field`."""

    dest: str
    record: str
    field: str

    def lets(self):
        return [self._let(self.dest, f"{self.record}.{self.field}", self.record)]


# -- programs ------------------------------------------------------------------------------

@dataclass
class Program:
    """A Lean definition: its docstring, signature, nodes, and result registers. `preamble`
    is Lean text emitted before it (the structure that a round's state is declared as)."""

    name: str
    doc: str
    signature: str
    nodes: list
    results: list
    preamble: str = ""

    def bindings(self):
        """Every `let`, with the node it belongs to."""
        return [(node, let) for node in self.nodes for let in node.lets()]

    def liveness(self):
        """For each binding of `bindings()`, whether a later binding or the result reads it."""
        bindings = self.bindings()
        needed = set(self.results)
        live = [False] * len(bindings)
        for i in reversed(range(len(bindings))):
            let = bindings[i][1]
            if let.name in needed:
                live[i] = True
                needed.discard(let.name)
                needed |= let.reads
        return live

    def live_bindings(self):
        return [b for b, keep in zip(self.bindings(), self.liveness(), strict=True) if keep]
