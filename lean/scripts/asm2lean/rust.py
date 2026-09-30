"""Front end: the `asm!` block of a Rust function.

A block is its template strings, one instruction each, and its operand declarations. The
template placeholders `{name}` are the operands' names, which the lifter treats as register
names: the register allocation the compiler chooses is not modelled. Before the block, the
function may copy argument limbs into the variables that `inout` operands name, either one by
one (`let mut a0 = value[0];`) or by destructuring (`let [mut r0, ...] = *lhs;`).
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Operand:
    """An operand declaration `name = direction(reg) expr`."""

    name: str
    direction: str  # "in", "out", or "inout"
    expr: str


@dataclass(frozen=True)
class AsmBlock:
    """The `asm!` block of one function."""

    function: str
    templates: tuple  # instruction texts, placeholders replaced by operand names
    operands: tuple  # Operand, in declaration order
    copies: dict  # variable -> (argument, limb index), from the bindings before the block

    @classmethod
    def parse(cls, source, function):
        found = re.search(rf"^\s*pub(?:\([^)]*\))? fn {function}\(", source, re.MULTILINE)
        if found is None:
            raise ValueError(f"no `fn {function}(`")
        text = source[found.start():]
        text = text[:text.index("options(")]

        copies = {var: (arg, int(i))
                  for var, arg, i in re.findall(r"let mut (\w+) = (\w+)\[(\d)\];", text)}
        for names, arg in re.findall(r"let \[((?:mut \w+, )*mut \w+)\] = \*(\w+);", text):
            for i, var in enumerate(re.findall(r"mut (\w+)", names)):
                copies[var] = (arg, i)

        templates = []
        for template in re.findall(r'^\s*"([^"]*)",', text, re.MULTILINE):
            line = re.sub(r"\{(\w+)\}", r"\1", template)
            templates.append(re.sub(r"\s+", " ", line.replace(", ", ",")).strip())

        operands = tuple(Operand(*d) for d in
                         re.findall(r'^\s*(\w+) = (in|out|inout)\(reg\) ([^,]+),', text,
                                    re.MULTILINE))
        return cls(function, tuple(templates), operands, copies)

    def limb(self, expr):
        """The argument limb `(argument, index)` that an input expression reads, if any:
        `lhs[0]` directly, or a variable copied from one before the block."""
        m = re.fullmatch(r"(\w+)\[(\d)\]", expr)
        if m:
            return m.group(1), int(m.group(2))
        return self.copies.get(expr)
