"""The pipeline, and the command line shared by the projects that use it."""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from . import lean
from .lift import lift
from .reroll import reroll
from .rust import AsmBlock
from .skeleton import check_spec, skeleton
from .vectors import examples


@dataclass(frozen=True)
class LeanModule:
    """A generated Lean file: its path, copyright header, imports, docstring, and namespace."""

    path: Path
    header: str
    imports: tuple
    doc: str
    namespace: str

    def text(self, body):
        return lean.module(self.header, self.imports, self.doc, self.namespace, body)


@dataclass(frozen=True)
class Transcription:
    """The blocks of one Rust file, lifted by one instruction set, into one Lean module."""

    source: Path
    lifter: type
    scalars: dict  # scalar argument -> the hypothesis bounding it
    blocks: tuple
    module: LeanModule

    def programs(self):
        source = self.source.read_text()
        programs = []
        for block in self.blocks:
            program, instructions = lift(AsmBlock.parse(source, block.function), block,
                                         self.lifter, self.scalars)
            if block.loop is None:
                programs.append(program)
            else:
                programs += reroll(program, instructions, block.loop, block.args)
        return programs


@dataclass(frozen=True)
class Vectors:
    """A vectors file, checked as examples in one Lean module."""

    source: Path
    fields: dict
    operations: dict
    module: LeanModule


def main(transcriptions, vectors, argv=None):
    parser = argparse.ArgumentParser(description="Generate the Lean transcription of the "
                                     "crate's `asm!` blocks, and check the proofs' skeletons.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skeleton", metavar="NAME",
                      help="print the generated part of the proof of NAME")
    mode.add_argument("--check-spec", metavar="FILE",
                      help="check that FILE contains every skeleton, annotations aside")
    args = parser.parse_args(argv)
    programs = [p for t in transcriptions for p in t.programs()]

    if args.skeleton:
        for program in programs:
            if program.name == args.skeleton:
                print("\n".join(skeleton(program)))
                return 0
        print(f"no routine {args.skeleton}", file=sys.stderr)
        return 1
    if args.check_spec:
        ok = check_spec(args.check_spec, programs)
        print(f"{args.check_spec}: skeletons {'current' if ok else 'STALE'}")
        return 0 if ok else 1

    written = []
    for t in transcriptions:
        t.module.path.write_text(t.module.text(lean.definitions(t.programs()) + "\n"))
        written.append(t.module.path)
    for v in vectors:
        body = examples(v.source.read_text().splitlines(), v.fields, v.operations)
        v.module.path.write_text(v.module.text(body))
        written.append(v.module.path)
    print(f"wrote {' and '.join(map(str, written))}")
    return 0
