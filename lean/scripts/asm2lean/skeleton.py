"""Back end: the mechanical part of a program's correctness proof.

The skeleton is symbolic execution of the program inside a Lean proof. The theorem's hypothesis
`hr : r = <program> ...` is unfolded; then, one node at a time, the node's `let`s are extracted
from `hr` under fresh names, their defining equations are recorded (by `rfl`, in `%`/`/` form),
the locals are made opaque, and the facts that follow are derived: each register is below
`2^64`, each carry at most `1`, and each (result, carry) pair satisfies its linear equation.
Every fact is an instance of one lemma, and is linear, so the hand-written annotations can close
the proof with `omega`, in Presburger arithmetic.

Two refinements keep the facts linear. A `mul` and the `umulh` of the same operands give the two
words of one product, so their equations are combined into `lo + 2^64 * hi = a * b`. An `lsl`
by 62 and an `lsr` by 2 of the same register split it at bit 2, and combine alike. A `umulh`
whose low word is never computed gets a ghost low word.

Names: shadowed registers are renamed apart (`r1`, `r1_1`, ...), which is the program's static
single assignment form. An argument bound under its own name (`inv := inv`) is primed, so that
extracting it does not shadow the theorem's variable.

Extracting one node at a time (`extract_lets +onlyGivenNames`) keeps the rest of the chain
folded inside `hr`, so that `clear_value` has one hypothesis to revert and re-check. With every
let extracted up front, each `clear_value` re-checks all the later locals and equations, which
is quadratic in the chain's length and exhausted the heartbeat budget on the multiplication
routine's 264 locals.
"""
import re
import sys
from pathlib import Path

from .ir import (
    AddCarry,
    AddTrunc,
    Call,
    Load,
    Mov,
    MulHi,
    MulLo,
    Project,
    Scalar,
    Select,
    Shl,
    Shr,
    SubBorrow,
    SubFlag,
    is_literal,
)
from .lean import wrap_words

MOD_LT = "Nat.mod_lt _ (Nat.two_pow_pos _)"


def ssa_names(lets):
    """Unique names for `lets`: the first binding of a register keeps its name, later ones get
    `_1`, `_2`, ...; a binding `x := x` is primed."""
    counts, names = {}, []
    for let in lets:
        n = counts.get(let.name, 0)
        counts[let.name] = n + 1
        base = let.name + "'" if let.expr == let.name else let.name
        names.append(base if n == 0 else f"{base}_{n}")
    return names


def skeleton(program):
    """The skeleton of `program`, as lines."""
    return _Skeleton(program).lines()


class _Skeleton:

    def __init__(self, program):
        self.program = program
        self.current = {}  # register -> its current SSA name
        self.bound = {}  # SSA name -> the fact bounding it below 2^64 (or a carry by 1)
        self.narrow = set()  # `lsr` results, bounded below 2^(64 - k)
        self.products = {}  # (a, b) -> the `mul` result awaiting the matching `umulh`
        self.shifts = {}  # (a, k) -> the `lsl` result awaiting the matching `lsr`

    def lines(self):
        name = self.program.name
        out = [f"  -- generated skeleton for `{name}`: do not edit between the annotations",
               f"  unfold {name} at hr", "  lift_lets at hr"]
        live = self.program.live_bindings()
        names = ssa_names([let for _, let in live])
        groups = []  # (node, [(let, ssa name)]), one per node with a live binding
        for (node, let), ssa in zip(live, names, strict=True):
            if groups and groups[-1][0] is node:
                groups[-1][1].append((let, ssa))
            else:
                groups.append((node, [(let, ssa)]))
        for node, lets in groups:
            out += self.step(node, [ssa for _, ssa in lets])
            for let, ssa in lets:
                self.current[let.name] = ssa
        out.append("  subst hr")
        return out

    # -- operand facts --------------------------------------------------------------------

    def ssa(self, operand):
        return self.current.get(operand, operand)

    def lt64(self, v):
        """A proof that the value `v` is below 2^64."""
        if is_literal(v):
            return "(by decide)"
        if v in self.narrow:
            return f"(lt_of_lt_of_le {self.bound[v]} (by norm_num))"
        return self.bound[v]

    def le1(self, v):
        """A proof that the carry `v` is at most 1."""
        return "(by decide)" if is_literal(v) else self.bound[v]

    # -- one node ---------------------------------------------------------------------------

    def step(self, node, group):
        """The lines for one node whose live bindings get the names `group`: extract them,
        record their equations, make them opaque, and derive their facts."""
        equations, facts = [], []

        def eq(name, rhs):
            equations.append(f"  have e_{name} : {name} = {rhs} := rfl")

        def bounded(name, proof):
            facts.append(f"  have b_{name} : {name} < 2^64 := by rw [e_{name}]; {proof}")
            self.bound[name] = f"b_{name}"

        label = group[1] if isinstance(node, (AddCarry, SubBorrow)) else group[0]
        match node:
            case Load(record=record, field=f):
                eq(group[0], f"{record.name}.{f}")
                bounded(group[0], f"exact {record.bound}.{record.projection(f)}")
            case Scalar(name=name, bound=hyp):
                eq(group[0], name)
                bounded(group[0], f"exact {hyp}")
            case Mov(src=src):
                a = self.ssa(src)
                eq(group[0], a)
                bounded(group[0], "decide" if is_literal(a) else f"exact {self.lt64(a)}")
            case MulLo():
                nm, a, b = group[0], self.ssa(node.a), self.ssa(node.b)
                eq(nm, f"{a} * {b} % 2^64")
                bounded(nm, f"exact {MOD_LT}")
                self.products[(a, b)] = nm  # its `%` equation is cleared at the matching `umulh`
            case MulHi():
                facts += self.mul_hi(eq, group[0], self.ssa(node.a), self.ssa(node.b))
            case Shl(k=k):
                nm, a = group[0], self.ssa(node.a)
                eq(nm, f"{a} * 2^{k} % 2^64")
                bounded(nm, f"exact {MOD_LT}")
                self.shifts[(a, k)] = nm
            case Shr(k=k):
                facts += self.shift_right(eq, group[0], self.ssa(node.a), k)
            case AddCarry() | SubBorrow():
                facts += self.carry_op(eq, node, group)
            case AddTrunc():
                facts += self.add_trunc(eq, group[0], *map(self.ssa, (node.a, node.b, node.cin)))
            case SubFlag():
                facts += self.sub_flag(eq, group[0], *map(self.ssa, (node.a, node.b, node.cin)))
            case Select():
                nm, c = group[0], self.ssa(node.flag)
                x, y = self.ssa(node.if_zero), self.ssa(node.if_nonzero)
                eq(nm, f"(if {c} = 0 then {x} else {y})")
                facts += [f"  have b_{nm} : {nm} < 2^64 := by",
                          f"    rw [e_{nm}]; split <;> first | exact {self.lt64(x)} | "
                          f"exact {self.lt64(y)}"]
                self.bound[nm] = f"b_{nm}"
            case Call():
                eq(group[0], node.expr(self.ssa))
            case Project(record=record, field=f):
                eq(group[0], f"{self.ssa(record)}.{f}")
                # The bound is supplied by the annotation that applies the callee's theorem.
                self.bound[group[0]] = f"b_{group[0]}"
            case _:
                raise ValueError(f"no skeleton step for {node}")

        return [f"  -- {label}: {node.comment}",
                *wrap_words("extract_lets +onlyGivenNames", group, " at hr"),
                *equations,
                f"  clear_value {' '.join(group)}",
                *facts]

    def mul_hi(self, eq, nm, a, b):
        eq(nm, f"{a} * {b} / 2^64")
        facts = [f"  have p_{nm} : {a} * {b} < 2^64 * 2^64 := "
                 f"Nat.mul_lt_mul'' {self.lt64(a)} {self.lt64(b)}",
                 f"  have b_{nm} : {nm} < 2^64 := by rw [e_{nm}]; "
                 f"exact Nat.div_lt_of_lt_mul p_{nm}"]
        self.bound[nm] = f"b_{nm}"
        if (a, b) in self.products:
            lo = self.products.pop((a, b))
            facts += [f"  have d_{nm} : {lo} + 2^64 * {nm} = {a} * {b} := by",
                      f"    rw [e_{lo}, e_{nm}]; exact Nat.mod_add_div _ _",
                      f"  clear e_{lo} e_{nm}"]
        else:
            # The low word is never computed (its cancellation is arranged by `subs`); name it
            # as a ghost so that later steps need no `%`.
            facts += [f"  obtain ⟨lo_{nm}, b_lo_{nm}, d_{nm}⟩ :",
                      f"      ∃ lo, lo < 2^64 ∧ lo + 2^64 * {nm} = {a} * {b} :=",
                      f"    ⟨{a} * {b} % 2^64, {MOD_LT},",
                      f"      by rw [e_{nm}]; exact Nat.mod_add_div _ _⟩",
                      f"  clear e_{nm}"]
        return facts

    def shift_right(self, eq, nm, a, k):
        eq(nm, f"{a} / 2^{k}")
        facts = [f"  have b_{nm} : {nm} < 2^{64 - k} := by",
                 f"    rw [e_{nm}]; exact Nat.div_lt_of_lt_mul "
                 f"(lt_of_lt_of_eq {self.lt64(a)} (by norm_num))"]
        self.bound[nm] = f"b_{nm}"
        self.narrow.add(nm)
        if (a, 64 - k) in self.shifts:
            lo = self.shifts.pop((a, 64 - k))
            if k != 2:
                raise ValueError(f"lsl/lsr split by {64 - k}/{k}: "
                                 "add a lemma to the spec preamble")
            facts += [f"  have sh_{nm} : {lo} + 2^64 * {nm} = {a} * 2^{64 - k} := by",
                      f"    rw [e_{lo}, e_{nm}]; exact lsl62_lsr2_split _",
                      f"  clear e_{lo} e_{nm}"]
        return facts

    def carry_op(self, eq, node, group):
        """An `addc` or `subc` whose pair, result, and (when live) carry are `group`. A carry
        that nothing reads is not bound; it is obtained as a ghost."""
        a, b, cin = map(self.ssa, (node.a, node.b, node.cin))
        xn = group[1]
        dead_carry = len(group) == 2
        cn = f"k_{xn}" if dead_carry else group[2]
        if isinstance(node, AddCarry):
            value = f"({a} + {b} + {cin})"
            linear = f"{xn} + 2^64 * {cn} = {a} + {b} + {cin}"
            linear_proof = "Nat.mod_add_div _ _"
            carry_proof = (f"addc_carry_le_one {a} {b} {cin} {self.lt64(a)} {self.lt64(b)} "
                           f"{self.le1(cin)}")
        else:
            value = f"({a} + 2^64 - {b} - (1 - {cin}))"
            linear = f"{xn} + 2^64 * {cn} + {b} + 1 = {a} + 2^64 + {cin}"
            linear_proof = f"subc_lin {a} {b} {cin} {self.lt64(b)} {self.le1(cin)}"
            carry_proof = f"subc_carry_le_one {a} {b} {cin} {self.lt64(a)}"
        eq(xn, f"{value} % 2^64")
        self.bound[xn] = f"b_{xn}"
        if dead_carry:
            return [f"  have b_{xn} : {xn} < 2^64 := by rw [e_{xn}]; exact {MOD_LT}",
                    f"  obtain ⟨{cn}, b_{cn}, l_{xn}⟩ :",
                    f"      ∃ k, k ≤ 1 ∧ {linear.replace(cn, 'k')} :=",
                    f"    ⟨{value} / 2^64, {carry_proof},",
                    f"      by rw [e_{xn}]; exact {linear_proof}⟩",
                    f"  clear e_{xn}"]
        eq(cn, f"{value} / 2^64")
        self.bound[cn] = f"b_{cn}"
        return [f"  have l_{xn} : {linear} := by",
                f"    rw [e_{xn}, e_{cn}]; exact {linear_proof}",
                f"  have b_{xn} : {xn} < 2^64 := by rw [e_{xn}]; exact {MOD_LT}",
                f"  have b_{cn} : {cn} ≤ 1 := by",
                f"    rw [e_{cn}]; exact {carry_proof}",
                f"  clear e_{xn} e_{cn}"]

    def add_trunc(self, eq, nm, a, b, cin):
        eq(nm, f"({a} + {b} + {cin}) % 2^64")
        self.bound[nm] = f"b_{nm}"
        return [f"  have b_{nm} : {nm} < 2^64 := by rw [e_{nm}]; exact {MOD_LT}",
                f"  obtain ⟨k_{nm}, b_k_{nm}, l_{nm}⟩ :",
                f"      ∃ k, k ≤ 1 ∧ {nm} + 2^64 * k = {a} + {b} + {cin} :=",
                f"    ⟨({a} + {b} + {cin}) / 2^64, addc_carry_le_one {a} {b} {cin} "
                f"{self.lt64(a)} {self.lt64(b)} {self.le1(cin)},",
                f"      by rw [e_{nm}]; exact Nat.mod_add_div _ _⟩",
                f"  clear e_{nm}"]

    def sub_flag(self, eq, nm, a, b, cin):
        eq(nm, f"({a} + 2^64 - {b} - (1 - {cin})) / 2^64")
        self.bound[nm] = f"b_{nm}"
        return [f"  have b_{nm} : {nm} ≤ 1 := by rw [e_{nm}]; "
                f"exact subc_carry_le_one {a} {b} {cin} {self.lt64(a)}",
                f"  have l_{nm} : ({nm} = 1 ∧ {b} + 1 ≤ {a} + {cin}) ∨ "
                f"({nm} = 0 ∧ {a} + {cin} < {b} + 1) :=",
                f"    subc_carry_cases {a} {b} {cin} _ e_{nm} {self.lt64(a)} {self.lt64(b)} "
                f"{self.le1(cin)}",
                f"  clear e_{nm}"]


def check_spec(path, programs):
    """Whether each program's skeleton appears verbatim and contiguously in the file at `path`
    once its `-- BEGIN ... -- END` annotation blocks are removed and blank lines dropped. A
    program whose skeleton does not start in the file is not proved there, and is skipped."""
    text = Path(path).read_text()
    stripped = re.sub(r"(?ms)^\s*-- BEGIN[^\n]*\n.*?^\s*-- END[^\n]*\n", "", text)
    remaining = [line for line in stripped.splitlines() if line.strip()]
    ok = True
    for program in programs:
        expected = [line for line in skeleton(program) if line.strip()]
        n = len(expected)
        if expected[1] not in remaining:  # `unfold <program> at hr`
            continue
        for start in range(len(remaining) - n + 1):
            if remaining[start:start + n] == expected:
                del remaining[start:start + n]
                break
        else:
            _report_divergence(path, program.name, expected, remaining)
            ok = False
    return ok


def _report_divergence(path, name, expected, remaining):
    i = remaining.index(expected[1])
    for j, line in enumerate(expected):
        found = remaining[i + j] if i + j < len(remaining) else "<eof>"
        if found != line:
            print(f"{path}: skeleton of {name} diverges at skeleton line {j}:", file=sys.stderr)
            print(f"  expected: {line}", file=sys.stderr)
            print(f"  found:    {found}", file=sys.stderr)
            return
