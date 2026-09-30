"""Loop rerolling: folding an unrolled loop back into a function called once per round.

Some blocks were generated as a prologue, one loop body repeated a fixed number of times, and an
epilogue: as a word `u v_1 ... v_n w` over instructions. The rounds must be the same body up to
renaming one operand (the round's `rhs` limb), i.e. alpha-equivalent; that is checked on the
instructions. The body is then transcribed once, as a function of the registers that cross its
boundary, and the block becomes `u` followed by one call per round, then `w`.

The registers crossing the boundary come from the same dataflow facts as liveness: the body's
inputs are the registers it reads before writing them (its upward-exposed uses), and its outputs
are the registers it writes that the code after it reads before writing them again. An input
whose value on entry is an argument of the block stays a plain argument of the round function;
the others are the state carried from round to round, which must be exactly the `roles` the
loop declares.
"""
from dataclasses import dataclass

from .ir import Call, Load, Program, Project, Record, Scalar
from .lean import state_structure


@dataclass(frozen=True)
class Role:
    """A register carried from one round to the next, and its field in the state record."""

    register: str
    field: str
    doc: str


@dataclass(frozen=True)
class Loop:
    """An unrolled loop in a block: `end` is the instruction that ends the prologue and each
    round, `count` the number of rounds, `operand` the register that differs between rounds
    (`{i}` is the round number), taken as the parameter `param`. The round function is `round`,
    with docstring `round_doc`; its state record `state` is passed as the argument `state_arg`,
    has the fields of `roles`, of which `value` make up the accumulator, and is declared in the
    transcription when `emit_struct` is set. `param_bound` and `state_bound` are the hypotheses
    that bound the parameter and the state."""

    end: str
    count: int
    operand: str
    round: str
    round_doc: str
    state: str
    state_arg: str
    roles: tuple
    value: tuple
    param: str
    param_bound: str
    state_bound: str
    emit_struct: bool = True

    def register(self, k):
        return self.operand.format(i=k)


def reroll(program, instructions, loop, args):
    """The round function and the rerolled block, from the block's `program`."""
    name = program.name
    ends = [pc for pc, ins in enumerate(instructions) if ins.text == loop.end]
    if len(ends) != loop.count + 1:
        raise ValueError(f"{name}: expected {loop.count + 1} `{loop.end}`, found {len(ends)}")
    _check_rounds_alpha_equivalent(name, instructions, ends, loop)

    def segment(node):
        """0 for the prologue, k for round k, count + 1 for the epilogue."""
        if node.pc is None or node.pc <= ends[0]:
            return 0
        return next((k for k in range(1, loop.count + 1) if node.pc <= ends[k]), loop.count + 1)

    prologue = [n for n in program.nodes if segment(n) == 0]
    body = [n for n in program.nodes if segment(n) == 1]
    after = [n for n in program.nodes if segment(n) > 1]
    epilogue = [n for n in program.nodes if segment(n) == loop.count + 1]

    param_register = loop.register(1)
    inputs = _upward_exposed(body, exclude={param_register})
    written = {let.name for n in body for let in n.lets()}
    outputs = [r for r in _upward_exposed(after) if r in written]
    registers = [role.register for role in loop.roles]
    if sorted(outputs) != sorted(registers):
        raise ValueError(f"{name}: registers carried between rounds are {sorted(outputs)}, "
                         f"`roles` lists {sorted(registers)}")

    # An input is a plain argument when the prologue binds it to one; otherwise it is state.
    defined = {let.name: n for n in prologue for let in n.lets()}
    missing = [r for r in inputs if r not in defined]
    if missing:
        raise ValueError(f"{name}: the round reads {missing}, which the prologue never writes")
    arguments = [(r, defined[r]) for r in inputs if isinstance(defined[r], (Load, Scalar))]
    carried = [r for r in inputs if not isinstance(defined[r], (Load, Scalar))]
    if sorted(carried) != sorted(registers):
        raise ValueError(f"{name}: registers read from the previous round are "
                         f"{sorted(carried)}, `roles` lists {sorted(registers)}")

    limb_args = [a.name for a in args
                 if any(isinstance(n, Load) and n.record.name == a.name for _, n in arguments)]
    scalars = [r for r, n in arguments if isinstance(n, Scalar)]
    state = Record(loop.state_arg, loop.state, tuple(role.field for role in loop.roles),
                   loop.state_bound)

    def order(argument):
        n = argument[1]
        return (0, limb_args.index(n.record.name), n.field) if isinstance(n, Load) else (1,)

    round_nodes = ([n for _, n in sorted(arguments, key=order)]
                   + [Scalar(comment="argument", dest=param_register, name=loop.param,
                             bound=loop.param_bound)]
                   + [Load(comment="argument", dest=role.register, record=state, field=role.field)
                      for role in loop.roles]
                   + body)
    signature = (f"def {loop.round} ({' '.join(limb_args)} : Limbs) "
                 f"({'inv ' if scalars else ''}{loop.param} : Nat) "
                 f"({loop.state_arg} : {loop.state}) : {loop.state} :=")
    preamble = state_structure(name, loop) if loop.emit_struct else ""
    round_program = Program(loop.round, loop.round_doc, signature, round_nodes, registers,
                             preamble)

    calls = []
    for k in range(1, loop.count + 1):
        result = f"round{k}"
        calls.append(Call(comment=f"round {k}", dest=result, function=loop.round,
                          fixed=tuple(limb_args), scalars=(*scalars, loop.register(k)),
                          state=tuple(registers)))
        calls += [Project(comment=f"round {k} output", dest=role.register, record=result,
                          field=role.field) for role in loop.roles]
    block_program = Program(name, program.doc, program.signature, prologue + calls + epilogue,
                            program.results)
    return [round_program, block_program]


def _check_rounds_alpha_equivalent(name, instructions, ends, loop):
    first = loop.register(1)

    def round_k(k):
        rename = {loop.register(k): first}
        return [(ins.op, tuple(rename.get(t, t) for t in ins.operands))
                for ins in instructions[ends[k - 1] + 1:ends[k] + 1]]

    if any(round_k(k) != round_k(1) for k in range(2, loop.count + 1)):
        raise ValueError(f"{name}: the round bodies differ beyond `{first}`")


def _upward_exposed(nodes, exclude=()):
    """The registers that `nodes` read before writing them, in the order first read."""
    written, exposed = set(), []
    for node in nodes:
        for let in node.lets():
            for r in sorted(let.reads):
                if r not in written and r not in exposed and r not in exclude:
                    exposed.append(r)
            written.add(let.name)
    return exposed
