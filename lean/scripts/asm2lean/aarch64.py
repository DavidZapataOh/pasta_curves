"""The AArch64 instructions that the blocks use, lifted to the IR.

The carry flag is the register `c`. `xzr` reads as `0`, and an immediate `#n` as the literal
`n`. The flag conventions are AArch64's: `adds`/`subs` take carry-in `0`/`1`, `adcs`/`sbcs`
take the flag, and after a subtraction the flag is set when no borrow occurred, so `lo`/`cc`
select on a clear flag and `cs`/`hs` on a set one (see `PastaAArch64Asm.Semantics`).
"""
import re
from enum import Enum

from .ir import (
    AddCarry,
    AddTrunc,
    FlagState,
    Mov,
    MulHi,
    MulLo,
    Select,
    Shl,
    Shr,
    SubBorrow,
    SubFlag,
)
from .lift import Instruction, Lifter

FLAG = "c"
ZERO = "xzr"


class Mnemonic(str, Enum):
    MOV = "mov"
    MUL = "mul"
    UMULH = "umulh"
    LSL = "lsl"
    LSR = "lsr"
    ADDS = "adds"
    ADCS = "adcs"
    ADC = "adc"
    SUBS = "subs"
    SBCS = "sbcs"
    CSEL = "csel"


class Condition(str, Enum):
    LO = "lo"  # carry clear
    CC = "cc"  # carry clear
    CS = "cs"  # carry set
    HS = "hs"  # carry set

    @property
    def flag_state(self):
        return FlagState.CLEAR if self in (Condition.LO, Condition.CC) else FlagState.SET


def immediate(token):
    """An immediate such as `#62`, `#0x4000000000000000`, or `8*1`, as an integer."""
    s = token.lstrip("#")
    if re.fullmatch(r"0x[0-9a-fA-F]+", s):
        return int(s, 16)
    m = re.fullmatch(r"(-?[0-9]+)(?:\*([0-9]+))?", s)
    if m is None:
        raise ValueError(f"unexpected immediate {token}")
    return int(m.group(1)) * int(m.group(2) or 1)


class AArch64(Lifter):

    @staticmethod
    def parse(text):
        m = re.match(r"(\S+)\s*(.*)", text)
        try:
            op = Mnemonic(m.group(1))
        except ValueError:
            raise ValueError(f"unhandled instruction: {text}") from None
        return Instruction(op, tuple(re.findall(r"\[[^\]]*\]!?|[^,\s]+", m.group(2))), text)

    def value(self, token):
        """The value an operand token reads: a register, or a literal."""
        if token == ZERO:
            return "0"
        if token.startswith("#"):
            return str(immediate(token))
        return self.read(token)

    def lift(self, ins):
        op, t = ins.op, ins.operands
        if t and t[0] == ZERO and op not in (Mnemonic.SUBS, Mnemonic.SBCS):
            raise ValueError(f"unexpected write to {ZERO}: {ins.text}")
        match op:
            case Mnemonic.MOV:
                self.emit(Mov, dest=t[0], src=self.value(t[1]))
            case Mnemonic.MUL:
                self.emit(MulLo, dest=t[0], a=self.value(t[1]), b=self.value(t[2]))
            case Mnemonic.UMULH:
                self.emit(MulHi, dest=t[0], a=self.value(t[1]), b=self.value(t[2]))
            case Mnemonic.LSL:
                self.emit(Shl, dest=t[0], a=self.value(t[1]), k=immediate(t[2]))
            case Mnemonic.LSR:
                self.emit(Shr, dest=t[0], a=self.value(t[1]), k=immediate(t[2]))
            case Mnemonic.ADDS | Mnemonic.ADCS:
                cin = "0" if op is Mnemonic.ADDS else self.read(FLAG)
                self.emit(AddCarry, dest=t[0], carry=FLAG, a=self.value(t[1]),
                          b=self.value(t[2]), cin=cin)
            case Mnemonic.ADC:
                self.emit(AddTrunc, dest=t[0], a=self.value(t[1]), b=self.value(t[2]),
                          cin=self.read(FLAG))
            case Mnemonic.SUBS | Mnemonic.SBCS:
                cin = "1" if op is Mnemonic.SUBS else self.read(FLAG)
                a, b = self.value(t[1]), self.value(t[2])
                if t[0] == ZERO:
                    self.emit(SubFlag, carry=FLAG, a=a, b=b, cin=cin)
                else:
                    self.emit(SubBorrow, dest=t[0], carry=FLAG, a=a, b=b, cin=cin)
            case Mnemonic.CSEL:
                try:
                    condition = Condition(t[3])
                except ValueError:
                    raise ValueError(f"unexpected condition: {ins.text}") from None
                self.emit(Select, dest=t[0], flag=self.read(FLAG), x=self.value(t[1]),
                          y=self.value(t[2]), x_when=condition.flag_state)
