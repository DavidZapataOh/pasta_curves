"""Lifting: from an `asm!` block to a program of the IR.

The operand declarations bind the arguments: an `in` or `inout` operand reads an argument limb
or a scalar argument, and the named `out` operands and the `inout` operands are the results.
The instructions are then lifted one at a time by an instruction set's `Lifter`, which knows
their semantics. The lifter keeps the set of registers that hold a value, so that a register
read before any instruction or operand wrote it is reported rather than transcribed.
"""
from dataclasses import dataclass

from .ir import Load, Program, Scalar


@dataclass(frozen=True)
class Instruction:
    """One instruction: its mnemonic, its operand tokens, and its text as written."""

    op: str
    operands: tuple
    text: str


class Lifter:
    """The instruction-set-independent part of lifting. A subclass parses an instruction's
    text (`parse`) and emits the nodes of its semantics (`lift`)."""

    def __init__(self):
        self.nodes = []
        self.registers = set()  # registers that hold a value
        self.pc, self.text = None, "argument"

    @staticmethod
    def parse(text):
        raise NotImplementedError

    def lift(self, instruction):
        raise NotImplementedError

    def read(self, register):
        if register not in self.registers:
            raise ValueError(f"{register} read before being written")
        return register

    def emit(self, node_type, **fields):
        node = node_type(comment=self.text, pc=self.pc, **fields)
        self.nodes.append(node)
        self.registers.update(let.name for let in node.lets())
        return node


@dataclass(frozen=True)
class Block:
    """An `asm!` block to transcribe: the Rust function, the Lean name and docstring, the
    record arguments in signature order, and the loop to fold, if any (see `reroll.py`)."""

    function: str
    name: str
    doc: str
    args: tuple  # Record
    loop: object = None


def lift(asm, block, lifter_type, scalars):
    """The program of `asm`, and its instructions. `scalars` maps the scalar arguments an
    operand may read to the hypotheses that bound them."""
    records = {r.name: r for r in block.args}
    lifter = lifter_type()
    outputs = []
    for operand in asm.operands:
        if operand.direction in ("in", "inout"):
            limb = asm.limb(operand.expr)
            if limb is not None:
                arg, i = limb
                if arg not in records:
                    raise ValueError(f"{block.name}: operand {operand.name} reads "
                                     f"{operand.expr}, not an argument")
                lifter.emit(Load, dest=operand.name, record=records[arg], field=f"l{i}")
                if operand.direction == "inout":
                    outputs.append((operand.name, operand.name))
            elif operand.expr in scalars:
                lifter.emit(Scalar, dest=operand.name, name=operand.expr,
                            bound=scalars[operand.expr])
            else:
                raise ValueError(f"{block.name}: unexpected input operand "
                                 f"{operand.name} = {operand.expr}")
        elif operand.direction == "out":
            if operand.expr != "_":
                outputs.append((operand.expr, operand.name))
        else:
            raise ValueError(f"{block.name}: unsupported operand direction {operand.direction}")

    instructions = [lifter.parse(text) for text in asm.templates]
    for pc, instruction in enumerate(instructions):
        lifter.pc, lifter.text = pc, instruction.text
        lifter.lift(instruction)
    results = [lifter.read(register) for _, register in sorted(outputs)]
    if len(results) != 4:
        raise ValueError(f"{block.name}: {len(results)} output operands")

    used = [n for n in lifter.nodes if isinstance(n, Scalar)]
    signature = (f"def {block.name} ({' '.join(r.name for r in block.args)} : "
                 f"{block.args[0].type})"
                 + "".join(f" ({n.name} : Nat)" for n in used)
                 + f" : {block.args[0].type} :=")
    return Program(block.name, block.doc, signature, lifter.nodes, results), instructions
