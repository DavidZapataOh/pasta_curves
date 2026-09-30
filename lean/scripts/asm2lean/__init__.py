"""A compiler from inline assembly in Rust to Lean definitions and their proof skeletons.

The compiler is organised as usual, as a pipeline over one intermediate representation:

    rust.py      front end: the `asm!` blocks of a Rust file, as instruction templates and
                 operand declarations
    lift.py      lifting: the instructions, through the semantics of one instruction set
                 (aarch64.py), to the IR; the operand declarations bind the arguments
    ir.py        the IR: a straight-line program in A-normal form over the primitives of the
                 Lean semantics, and liveness on it
    reroll.py    a pass that folds a repeated loop body back into a function called per round
    lean.py      back end: the Lean text of each program
    skeleton.py  back end: the mechanical part of each program's correctness proof
    vectors.py   the reference vectors, as kernel-checked Lean examples

`driver.py` composes the stages. A project describes its blocks, its Lean modules, and its
vectors, and calls the driver (see `lean/scripts/gen.py`).

To support another instruction set, add a lifter beside `aarch64.py`: the IR, the passes, and
both back ends are independent of the instruction set.
"""
