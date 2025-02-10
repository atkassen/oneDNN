################################################################################
# Copyright 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import enum
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class LicenseState(enum.Enum):
    NOT_SEEN = 0
    INSIDE = 1
    SEEN = 2


class FileStatus(enum.Enum):
    OK = 32
    FAIL = 31
    WARN = 33

    def color(self, buffer):
        fmtd = self.name.center(4, " ")
        if buffer.isatty():
            fmtd = f"\033[{self.value};1m{fmtd}\033[0m"  # ]]
        return fmtd


@dataclass
class Options:
    inplace: bool = False
    closing_comment: bool = False
    verbose: bool = False


def ignore(relpath):
    ignored_paths = [
        "src/cpu/ppc64",
        "src/cpu/s390x",
        "src/cpu/rv64",
    ]
    for ignored_path in ignored_paths:
        if relpath.startswith(ignored_path):
            return True
    return False


def get_file_guard(path):
    if path.startswith("src/gpu/intel/jit/gemm"):
        base = os.path.basename(path)
        if path != "src/gpu/intel/jit/gemm/" + base:
            path = "src/gemmstone_guard/" + os.path.basename(path)
    elif path.startswith("src/gpu/intel/microkernels"):
        path = path.replace("intel/", "")
    guard = path
    for c in "/.":
        guard = guard.replace(c, "_")
    return guard.split("_", 1)[1].upper()


@dataclass
class Directive:
    kind: str
    args: Optional[str]
    start: int
    end: int


@dataclass
class Block:
    open: Directive
    close: Directive
    children: List["Block"] = field(default_factory=list)


def ifndef_argument(directive: Directive):
    if directive.kind != "ifndef":
        return None
    if directive.args is None:
        return None
    return directive.args.split(None, 1)[0]


def find_guard_blocks(guard: str, root: List[Block]):
    guard_blocks = []
    alternate = None
    if len(root) == 1 and root[0].open.kind == "ifndef":
        alternate = root[0]
    for block in root:
        arg = ifndef_argument(block.open)
        if arg == guard:
            guard_blocks.append(block)
            continue
        blocks, alt = find_guard_blocks(guard, block.children)
        guard_blocks += blocks
        if len(root) == 1 and alternate is None:
            alternate = alt
    return guard_blocks, alternate


def insert_guard(lines: List[str], guard: str, line: int, add_comment: bool):
    close = "#endif"
    if add_comment:
        close += f" // {guard}"
    extra_lines = [close]
    lines[line:line] = ["", f"#ifndef {guard}", f"#define {guard}"]
    if lines[-1].startswith("// vim:"):
        extra_lines += ["", lines[-1]]
        lines = lines[:-1]
    if lines[-1].strip():
        lines.append("")  # Insert blank line before #endif
    lines += extra_lines + [""]  # force EOL
    return "\n".join(lines)


def no_code(lines):
    is_multiline_comment = False
    for *_, line in continuations(lines):
        line = line.strip()
        while line:
            if is_multiline_comment:
                if "*/" not in line:
                    break
                line = line.split("*/", 1)[1].lstrip()
                is_multiline_comment = False
            if line.startswith("/*"):
                is_multiline_comment = True
                continue
            if line.startswith("//"):
                break
            if line.startswith("#"):
                break
            return False
    return True


def insert_define(lines: List[str], guard: str, open: Directive):
    index = open.end + 1
    lines[index:index] = [f"#define {guard}"]
    return "\n".join(lines)


def continuations(lines):
    continuation = []
    for index, line in enumerate(lines):
        continuation.append(line)
        if line and line[-1] == "\\":
            continue
        combined = "\n".join(continuation)
        start = index + 1 - len(continuation)
        yield start, index, combined
        continuation = []


def fix_file(file, options):
    fullpath = os.path.realpath(file)
    _, ext = os.path.splitext(fullpath)
    if ext not in (".hpp", ".h"):
        return 0
    path_segments = "src/", "include/"
    for segment in path_segments:
        if segment not in fullpath:
            continue
        tail = fullpath.split(segment, 1)[1]
        relpath = segment + tail
        break
    else:
        return 0
    if ignore(relpath):
        return 0
    guard = get_file_guard(relpath)
    status, message = adjust_content(fullpath, guard)
    if status is not FileStatus.OK or options.verbose:
        print(f"[{status.color(sys.stdout)}] {relpath} ({message})")
    return status is FileStatus.FAIL


def adjust_content(fullpath: str, guard: str):
    with open(fullpath) as fd:
        content = fd.read()

    state = LicenseState.NOT_SEEN
    license_ends = 0
    lines = content.splitlines()
    if_stack: List[Directive] = []
    blocks: List[List[Block]] = [[]]
    defines: Dict[str, List[Directive]] = defaultdict(list)
    for start, end, raw_line in continuations(lines):
        line = raw_line.replace("\\\n", "").strip()
        if line.startswith("/*") and state == LicenseState.NOT_SEEN:
            state = LicenseState.INSIDE
            continue
        elif not line.startswith("*") and state == LicenseState.INSIDE:
            state = LicenseState.SEEN
            license_ends = end
        if not line.startswith("#"):
            continue
        rest = line[1:]
        kind, *rest = line[1:].split(None, 1)
        args = rest[0] if rest else None
        directive = Directive(kind, args, start, end)

        if kind == "endif":
            try:
                if_directive = if_stack.pop()
                children = blocks.pop()
            except IndexError:
                return FileStatus.FAIL, "mismatched #ifs/#endifs"
            block = Block(if_directive, directive, children)
            blocks[-1].append(block)
        elif kind == "pragma":
            if args is None or args.strip() != "once":
                continue
            block = Block(directive, directive, [])
            blocks[-1].append(block)
        elif kind in ("if", "ifdef", "ifndef"):
            blocks.append([])
            if_stack.append(directive)
        elif kind == "define":
            if args is None:
                continue
            name = args.split(None, 1)[0]
            defines[name].append(directive)

    if len(blocks) != 1:
        return FileStatus.FAIL, "mismatched #ifs/#endifs"
    force_write = False
    for block in blocks[0]:
        if block.open.kind == "pragma":
            lines[block.open.start : block.close.end + 1] = []
            force_write = options.inplace
            continue
        if block.open.kind != "ifndef":
            continue

    guards, root = find_guard_blocks(guard, blocks[0])
    if len(guards) == 1:
        if guard not in defines:
            content = insert_define(lines, guard, guards[0].open)
            if not options.inplace:
                return FileStatus.FAIL, "missing define"
            message = "added missing define"
        message = "correct guard found"
    elif len(guards) > 1:
        return FileStatus.FAIL, "too many guards"
    elif root is None:
        start = license_ends
        for block in blocks[0]:
            if block.open.start < start:
                continue
            section = lines[start : block.open.start]
            start = block.close.end
            if no_code(section):
                continue
            break
        else:
            if no_code(lines[start:]):
                return FileStatus.OK, "no content"
        content = insert_guard(
            lines,
            guard,
            license_ends,
            options.closing_comment,
        )
        message = "added missing guard"
        if not options.inplace:
            return FileStatus.FAIL, "missing guard"
    elif root.open.args is not None:
        old_guard = ifndef_argument(root.open)
        if old_guard is None:
            return FileStatus.FAIL, "broken guard"
        message = "found correct guard"
        if old_guard not in defines:
            content = insert_define(lines, old_guard, root.open)
            if not options.inplace:
                return FileStatus.FAIL, "missing define"
            message = "added missing define"
        if old_guard != guard:
            content = content.replace(old_guard, guard)
            if not options.inplace:
                return FileStatus.FAIL, f"wrong guard: {old_guard}"
            message = f"fixed incorrect guard {old_guard}"
    else:
        return FileStatus.FAIL, "broken top-level guard"
    if options.inplace or force_write:
        with open(fullpath, "w") as fd:
            fd.write(content)
    return FileStatus.OK, message


def find_files(basepath, options):
    fullpath = os.path.realpath(basepath)
    exit_code = 0
    if os.path.isfile(fullpath):
        return fix_file(fullpath, options)
    if not os.path.isdir(fullpath):
        return exit_code
    for dir, _, filenames in os.walk(fullpath):
        for filename in filenames:
            exit_code |= fix_file(os.path.join(dir, filename), options)
    return exit_code


def print_help(prog: str):
    print(
        f"""usage: {prog} [OPTIONS] files...

description:
    Checks the files (or directories) given for correct header guards in each
    .hpp or .h file.

options:
    -v    print passing in addition to failing files
    -i    modify files in-place
    -c    add a comment with the guard name after #endif (incomplete)
    -h    print this help and exit"""
    )


if __name__ == "__main__":
    exit_code = 0
    options = Options()
    args = sys.argv[1:]
    while args:
        if args[0] == "-i":
            options.inplace = True
        elif args[0] == "-c":
            options.closing_comment = True
        elif args[0] == "-v":
            options.verbose = True
        elif args[0] == "-h":
            print_help(sys.argv[0])
            sys.exit(0)
        else:
            break
        args.pop(0)

    if args[0] == "-i":
        inplace = True
        args = args[1:]
    for location in args:
        exit_code |= find_files(location, options)
    sys.exit(exit_code)
