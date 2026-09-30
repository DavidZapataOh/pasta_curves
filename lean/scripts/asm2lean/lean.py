"""Back end: the Lean text of programs and modules.

A program prints as a definition whose body is its live `let` bindings, each with its source
instruction as a trailing comment, aligned at one column shared by the whole module. A binding
that nothing reads is not printed; for an argument the block binds and never uses, a comment
records it.
"""
import textwrap

WIDTH = 100

# Code longer than this does not set the comment column (the round calls).
COMMENT_COLUMN_MAX = 40


def docstring(text):
    """A `/-- ... -/` docstring wrapped to WIDTH."""
    lines = textwrap.TextWrapper(width=WIDTH, break_long_words=False, break_on_hyphens=False,
                                 initial_indent="/-- ").wrap(text)
    if len(lines[-1]) + 3 <= WIDTH:
        lines[-1] += " -/"
    else:
        lines.append("-/")
    return "\n".join(lines)


def wrap_words(head, words, tail, indent="  "):
    """`head w1 w2 ... tail`, broken over lines at WIDTH with a 4-space continuation."""
    lines, current = [], indent + head
    for word in words:
        if len(current) + 1 + len(word) > WIDTH:
            lines.append(current)
            current = indent + "    " + word
        else:
            current += " " + word
    lines.append(current + tail)
    return lines


def module(header, imports, doc, namespace, body):
    """A Lean file: the copyright header, the imports, the module docstring, and `body` inside
    the namespace."""
    imports = "".join(f"import {i}\n" for i in imports)
    return f"{header}{imports}\n{doc}\n\nnamespace {namespace}\n\n{body}end {namespace}\n"


def body_lines(program):
    """The body of `program` as (code, comment) pairs; code is `None` for a comment line."""
    lines = []
    for (_, let), live in zip(program.bindings(), program.liveness(), strict=True):
        if live:
            lines.append((f"  let {let.name} := {let.expr}", let.comment))
        elif let.dead == "comment":
            lines.append((None, f"  -- {let.comment}: {let.name} = {let.expr} is never read"))
        elif let.dead == "error":
            raise ValueError(f"dead computation: {let.name} := {let.expr} ({let.comment})")
    return lines


def definitions(programs):
    """The definitions of `programs`, with one comment column for all of them: two spaces past
    the widest ordinary `let`."""
    bodies = [body_lines(p) for p in programs]
    column = 2 + max(len(code) for lines in bodies for code, _ in lines
                     if code is not None and len(code) <= COMMENT_COLUMN_MAX)
    return "\n".join(_definition(p, lines, column)
                     for p, lines in zip(programs, bodies, strict=True))


def _definition(program, lines, column):
    body = []
    for code, comment in lines:
        if code is None:
            body.append(comment)
        elif len(code) + 2 <= column:
            body.append(f"{code.ljust(column)}-- {comment}")
        else:
            body.append(f"{code}  -- {comment}")
    preamble = f"{program.preamble}\n" if program.preamble else ""
    return (f"{preamble}{docstring(program.doc)}\n{program.signature}\n" + "\n".join(body)
            + f"\n  ⟨{', '.join(program.results)}⟩\n")


def state_structure(owner, loop):
    """The structure of the registers that `owner`'s rounds carry, with `Bounded` and `toNat`."""
    state = loop.state
    fields = [role.field for role in loop.roles]
    lines = [f"/-- The registers that `{owner}` carries from one round to the next. -/",
             f"structure {state} where"]
    for role in loop.roles:
        lines += [f"  /-- {role.doc} -/", f"  {role.field} : Nat"]
    lines += ["  deriving DecidableEq, Repr", "", f"namespace {state}", "",
              "/-- Every field is below `2^64`. -/",
              f"def Bounded (s : {state}) : Prop :="]
    lines += wrap_words("", _joined([f"s.{f} < 2^64" for f in fields], " ∧"), "")
    weights = f"`2^0` to `2^{64 * (len(loop.value) - 1)}`"
    lines += ["", f"/-- The accumulator's value: limbs `{'`, `'.join(loop.value)}` with weights "
              f"{weights}. -/",
              f"def toNat (s : {state}) : Nat :="]
    terms = [f"{f'2^{64 * i} * ' if i else ''}s.{f}" for i, f in enumerate(loop.value)]
    lines += wrap_words("", _joined(terms, " +"), "")
    lines += ["", f"end {state}", ""]
    return "\n".join(lines)


def _joined(words, separator):
    """`words`, each but the last followed by `separator`."""
    return [w + separator for w in words[:-1]] + words[-1:]
