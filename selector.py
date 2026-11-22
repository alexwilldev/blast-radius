"""Blast radius selector.

For each patch in /data/commits, decide which sqlglot tests could fail and pack
them onto workers. Logs go to stderr; stdout carries one JSON line a commit.

Method (all maps are built from sqlglot at the pinned commit, never from patches):
- maps/coverage.*: per-test line coverage (pytest-cov contexts) of sqlglot/ and
  tests/, with the suite's process pools run in-process so their work is
  attributed to the test that submitted it.
- maps/fixtures.json: which tests open which fixture file.
- maps/import_lines.z: lines executed just by importing every module.
- maps/usage.*: which tests instantiate (or look up as a dialect) each class.
- /app/struct (built in the Dockerfile): def/class/statement spans per file and
  the static import graph.
A changed line selects the tests that execute it or its enclosing statement.
Code that runs at import time (module level, class bodies, signatures, new
definitions) cannot be attributed by coverage, so it falls back to every test
that executes any code of the module's dialect family and the modules that
import it. Edits whose AST is unchanged (comments, blank lines) run nothing.
Anything not understood runs every test.
"""

import ast
import json
import zlib
import os
import re
import sys
import types
import dis
from pathlib import Path

DATA = Path(os.environ.get("BR_DATA", "/data"))
SRC = Path(os.environ.get("BR_SRC", "/app/src"))  # sqlglot at eb11aec
APP = Path(os.environ.get("BR_APP", Path(__file__).resolve().parent))
STRUCT = Path(os.environ.get("BR_STRUCT", APP / "struct"))
MAPS = APP / "maps"
MAX_WORKERS = 8
IMPORT_CHECK_FANOUT = 8
STATE_CHECK_MIN_GAIN_MS = 3000
ZYGOTE_MAX_MODULES = 40  # re-import only this many modules (patched + importers) per state check  # run the import-state check only if it can save this much  # re-run a patched module's importers only when it has this few
WORKER_COST_MS = 2000


def log(*args):
    print(*args, file=sys.stderr)


# ---------------------------------------------------------------- diff parsing

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class FilePatch:
    __slots__ = ("old", "new", "hunks", "binary")

    def __init__(self):
        self.old = None  # path in the pinned tree, None if the file is new
        self.new = None  # path after the patch, None if deleted
        self.hunks = []  # (old_start, old_len, new_start, new_len, lines)
        self.binary = False

    @property
    def path(self):
        return self.old or self.new


def _strip_prefix(path):
    path = path.split("\t")[0]
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def parse_patch(text):
    files = []
    cur = None
    lines = text.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("diff --git "):
            cur = FilePatch()
            files.append(cur)
            m = re.match(r"diff --git a/(.*) b/(.*)$", line)
            if m:
                cur.old, cur.new = m.group(1), m.group(2)
            i += 1
            continue
        if cur is None and (line.startswith("--- ") or line.startswith("+++ ")):
            cur = FilePatch()
            files.append(cur)
        if cur is not None:
            if line.startswith("--- "):
                cur.old = _strip_prefix(line[4:])
                i += 1
                continue
            if line.startswith("+++ "):
                cur.new = _strip_prefix(line[4:])
                i += 1
                continue
            if line.startswith("new file mode"):
                cur.old = None
            elif line.startswith("deleted file mode"):
                cur.new = None
            elif line.startswith("rename from "):
                cur.old = line[len("rename from "):]
            elif line.startswith("rename to "):
                cur.new = line[len("rename to "):]
            elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
                cur.binary = True
            m = HUNK_RE.match(line)
            if m:
                os_, ol, ns, nl = m.groups()
                ol = 1 if ol is None else int(ol)
                nl = 1 if nl is None else int(nl)
                os_, ns = int(os_), int(ns)
                body = []
                i += 1
                seen_old = seen_new = 0
                while i < n and (seen_old < ol or seen_new < nl):
                    hl = lines[i]
                    if hl.startswith("\\"):
                        i += 1
                        continue
                    tag = hl[:1] or " "
                    if tag == " ":
                        seen_old += 1
                        seen_new += 1
                    elif tag == "-":
                        seen_old += 1
                    elif tag == "+":
                        seen_new += 1
                    else:
                        break
                    body.append(hl if hl else " ")
                    i += 1
                while i < n and lines[i].startswith("\\"):
                    i += 1
                cur.hunks.append((os_, ol, ns, nl, body))
                continue
        i += 1
    return files


def apply_hunks(old_text, hunks):
    """Return the new file text, or None if the hunks do not apply cleanly."""
    old = old_text.split("\n")
    out = []
    pos = 0  # 0-based index into old
    for os_, ol, _ns, _nl, body in hunks:
        start = os_ - 1 if ol else os_
        if start < pos:
            return None
        out.extend(old[pos:start])
        pos = start
        for hl in body:
            tag, content = hl[:1], hl[1:]
            if tag in (" ", "-"):
                if pos >= len(old) or old[pos] != content:
                    return None
                if tag == " ":
                    out.append(content)
                pos += 1
            else:
                out.append(content)
    out.extend(old[pos:])
    return "\n".join(out)


# ---------------------------------------------------------------- classification

FAMILY_DIRS = ("dialects", "parsers", "generators", "tokenizers", "typing")
IMPORT_SENSITIVE = (
    "tests/test_dialect_imports.py",
    "tests/test_dialect_entry_points.py",
    "tests/dialects/test_dialect.py::TestDialect::test_lazy_load",
)


class Unknown(Exception):
    """Something we do not understand: run everything."""


def _changed_lines(fp):
    for _os, _ol, _ns, _nl, body in fp.hunks:
        for hl in body:
            if hl[:1] in "+-":
                yield hl[1:]


def _added_lines(fp):
    for _os, _ol, _ns, _nl, body in fp.hunks:
        for hl in body:
            if hl[:1] == "+":
                yield hl[1:]


def old_lines_touched(fp, n):
    """Old-file line numbers removed, plus both neighbours of every insertion.

    Returns (touched, removed)."""
    out = set()
    removed = set()
    for os_, ol, _ns, _nl, body in fp.hunks:
        o = os_ if ol else os_ + 1  # line number of the next old line
        prev_tag = None
        for hl in body:
            tag = hl[:1]
            if tag == "-":
                out.add(o)
                removed.add(o)
                o += 1
            elif tag == "+":
                out.add(o - 1)
                out.add(o)
            else:
                o += 1
            prev_tag = tag
    return {x for x in out if 1 <= x <= n}, removed


class Maps:
    def __init__(self, ntests):
        self.ntests = ntests
        self.nbytes = (ntests + 7) // 8
        self._idx = None
        self._sets = None
        self._fix = None
        self._struct = {}
        self._imports = None
        self._file_union = {}
        self._set_cache = {}
        self._import_lines = None
        self._usage = None
        self._usage_sets = None

    @property
    def idx(self):
        if self._idx is None:
            self._idx = json.loads(zlib.decompress((MAPS / "coverage.idx").read_bytes()))
        return self._idx

    def bits(self, sid):
        b = self._set_cache.get(sid)
        if b is None:
            if self._sets is None:
                self._sets = (MAPS / "coverage.sets").read_bytes()
            nb = self.nbytes
            b = self._set_cache[sid] = int.from_bytes(self._sets[sid * nb:(sid + 1) * nb], "little")
        return b

    def line(self, path, line):
        sid = self.idx.get(path, {}).get(str(line))
        return 0 if sid is None else self.bits(sid)

    def lines(self, path, start, end):
        li = self.idx.get(path, {})
        out = 0
        for ln in range(start, end + 1):
            sid = li.get(str(ln))
            if sid is not None:
                out |= self.bits(sid)
        return out

    def file_union(self, path):
        if path not in self._file_union:
            out = 0
            for sid in set(self.idx.get(path, {}).values()):
                out |= self.bits(sid)
            self._file_union[path] = out
        return self._file_union[path]

    def struct(self, path):
        if path not in self._struct:
            p = STRUCT / (path + ".json")
            self._struct[path] = json.loads(p.read_text()) if p.exists() else None
        return self._struct[path]

    def class_users(self, path, qualname):
        """Tests that instantiated (or looked up as a dialect) this class or a subclass."""
        if self._usage is None:
            self._usage = json.loads(zlib.decompress((MAPS / "usage.idx").read_bytes()))
            self._usage_sets = (MAPS / "usage.sets").read_bytes()
        sid = self._usage.get(f"{path}::{qualname}")
        if sid is None:
            return 0
        nb = self.nbytes
        return int.from_bytes(self._usage_sets[sid * nb:(sid + 1) * nb], "little")

    def attr_readers(self, key):
        """Tests that read class attribute `key` ("sqlglot/x.py::Cls.ATTR"), from a
        recorder pass over the pinned suite; None if no recorder stood in for it."""
        if getattr(self, "_attrs", None) is None:
            try:
                self._attrs = json.loads(zlib.decompress((MAPS / "attrs.idx").read_bytes()))
                self._attrs_sets = (MAPS / "attrs.sets").read_bytes()
            except OSError:
                self._attrs = {}
        sid = self._attrs.get(key)
        if sid is None:
            return None
        nb = self.nbytes
        return int.from_bytes(self._attrs_sets[sid * nb:(sid + 1) * nb], "little")

    def import_lines(self, path):
        """Lines executed just by importing every module (no test running)."""
        if self._import_lines is None:
            self._import_lines = json.loads(zlib.decompress((MAPS / "import_lines.z").read_bytes()))
        return self._import_lines.get(path, ())

    @property
    def imports(self):
        if self._imports is None:
            self._imports = json.loads((STRUCT / "_imports.json").read_text())
        return self._imports

    @property
    def fixtures(self):
        if self._fix is None:
            self._fix = json.loads((MAPS / "fixtures.json").read_text())
        return self._fix


def _code_equal(a, b):
    """Same bytecode, names and constants, recursively; line numbers are ignored."""
    for attr in ("co_names", "co_varnames", "co_freevars", "co_cellvars", "co_flags",
                 "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_exceptiontable"):
        if getattr(a, attr, None) != getattr(b, attr, None):
            return False
    # Python 3.13+ class bodies store their own first line number
    # (`__firstlineno__ = <n>`): a comment above the class shifts only that value.
    # 3.13 loads it from co_consts (LOAD_CONST), later versions from the
    # instruction itself (LOAD_SMALL_INT); mask it in both places.
    skip_consts = set()
    if "__firstlineno__" in a.co_names:
        ia, ib = list(dis.get_instructions(a)), list(dis.get_instructions(b))
        if len(ia) != len(ib):
            return False
        for k, (x, y) in enumerate(zip(ia, ib)):
            if x.opname != y.opname:
                return False
            nxt = ia[k + 1] if k + 1 < len(ia) else None
            lineno_load = nxt is not None and nxt.opname == "STORE_NAME" and nxt.argval == "__firstlineno__"
            if lineno_load and x.opname == "LOAD_CONST":
                if x.arg != y.arg:
                    return False
                skip_consts.add(x.arg)
            elif x.arg != y.arg and not lineno_load:
                return False
    elif a.co_code != b.co_code:
        return False
    if len(a.co_consts) != len(b.co_consts):
        return False
    return all(i in skip_consts or _const_equal(p, q)
               for i, (p, q) in enumerate(zip(a.co_consts, b.co_consts)))


def _const_equal(x, y):
    """Compare compiled constants by value. Not by repr: a frozenset's repr order
    follows the per-process string hash seed, so equal sets can print differently."""
    if type(x) is not type(y):
        return False
    if isinstance(x, types.CodeType):
        return _code_equal(x, y)
    if isinstance(x, tuple):
        return len(x) == len(y) and all(_const_equal(p, q) for p, q in zip(x, y))
    if isinstance(x, frozenset):
        return x == y and sorted(map(repr, x)) == sorted(map(repr, y))
    return repr(x) == repr(y)


def _strip_docstrings(text):
    """Compiled code of `text` with every module, class and function docstring removed.
    The scored suite never reads docstrings (tests/test_docs.py is not in it)."""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                del body[0]
                if not body and not isinstance(node, ast.Module):
                    body.append(ast.Pass(lineno=node.lineno, col_offset=0))
    return compile(tree, "<doc>", "exec", dont_inherit=True)


def _in_triple_quotes(fp, old_text, new_text):
    """Cheap filter: does every changed line sit inside (or open/close) a triple-quoted string?"""
    old_lines, new_lines = old_text.splitlines(), new_text.splitlines()
    parity = {}

    def open_before(lines):
        # open_before[i]: is a triple-quoted string still open at the start of line i?
        key = id(lines)
        if key not in parity:
            out, d, s = [], 0, 0
            for line in lines:
                out.append(d % 2 == 1 or s % 2 == 1)
                d += line.count('"""')
                s += line.count("'''")
            parity[key] = out
        return parity[key]

    def inside(lines, idx):
        if not 0 <= idx < len(lines):
            return False
        line = lines[idx]
        return '"""' in line or "'''" in line or open_before(lines)[idx]

    for os_, _ol, ns, _nl, body in fp.hunks:
        o, n = os_ - 1, ns - 1
        for hl in body:
            tag = hl[:1]
            if tag == "-":
                if hl[1:].strip() and not inside(old_lines, o):
                    return False
                o += 1
            elif tag == "+":
                if hl[1:].strip() and not inside(new_lines, n):
                    return False
                n += 1
            else:
                o += 1
                n += 1
    return True


def _docstring_only(fp, old_text, new_text):
    if not _in_triple_quotes(fp, old_text, new_text):
        return False
    try:
        return _code_equal(_strip_docstrings(old_text), _strip_docstrings(new_text))
    except (SyntaxError, ValueError):
        return False


def _ast_equal(a, b):
    """Do the two sources compile to the same program (comments and blank lines aside)?"""
    try:
        ca = compile(a, "<old>", "exec", dont_inherit=True)
        cb = compile(b, "<new>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        return False
    return _code_equal(ca, cb)


def _def_names(text):
    """Qualified names of every def and class in a module (None if it does not parse);
    a name defined twice counts as new, since the second definition replaces the first."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    out, seen = set(), set()

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                out.add(name + ("#2" if name in seen else ""))
                seen.add(name)
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return out


def _def_headers(text):
    """{qualname: header of each def with its annotations left out}, or None unless the
    module has `from __future__ import annotations` (then annotations stay unevaluated
    strings, so an annotation-only edit of a signature changes no behaviour)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    if not any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names) for n in tree.body):
        return None
    out = {}

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                if not isinstance(child, ast.ClassDef):
                    a = child.args
                    args = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
                    extra = [x.arg if x else None for x in (a.vararg, a.kwarg)]
                    out[name] = ast.dump(ast.Tuple(
                        elts=[ast.Constant(type(child).__name__), ast.Constant(repr(args)),
                              ast.Constant(repr(extra)), ast.Constant(len(a.posonlyargs)),
                              ast.Constant(len(a.kwonlyargs))]
                        + child.decorator_list + a.defaults
                        + [d if d is not None else ast.Constant(None) for d in a.kw_defaults]
                        + list(getattr(child, "type_params", []) or []), ctx=ast.Load()))
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return out


_ENUM_BASES = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "AutoName"}
_enum_cache = {}


def _is_enum_class(name, depth=0):
    """Is `name` a class defined in sqlglot/ that derives from an Enum (found by source scan)?"""
    if name in _ENUM_BASES:
        return True
    if depth > 3:
        return False
    if name in _enum_cache:
        return _enum_cache[name]
    _enum_cache[name] = False
    pat = re.compile(r"^\s*class\s+" + re.escape(name) + r"\s*\(([^)]*)\)", re.M)
    found = []
    for f in sorted((SRC / "sqlglot").rglob("*.py")):
        try:
            txt = f.read_text()
        except OSError:
            continue
        if ("class " + name) not in txt:
            continue
        found += pat.findall(txt)
    ok = bool(found) and all(
        any(_is_enum_class(b.strip().split(".")[-1].split("[")[0], depth + 1) for b in bases.split(",") if b.strip())
        for bases in found
    )
    _enum_cache[name] = ok
    return ok


_CMP_SPLIT = re.compile(r"(\bis\s+not\b|\bis\b|==|!=)")
_SWAPS = {("is", "=="), ("==", "is"), ("is not", "!="), ("!=", "is not")}
_OPERAND = re.compile(r"[A-Za-z_][\w.]*")


def _singleton_text(op):
    """None, or Cls.MEMBER of an Enum class: `is` and `==` agree on these."""
    if op == "None":
        return True
    parts = op.split(".")
    return len(parts) >= 2 and parts[-1].isupper() and _is_enum_class(parts[-2])


def _is_eq_rewrite(fp):
    """Is every changed line the same as before except for `is`/`is not` <-> `==`/`!=`
    against None or an Enum member? Decided from the changed lines alone."""
    removed = [hl[1:] for _o, _l, _n, _m, body in fp.hunks for hl in body if hl[:1] == "-"]
    added = list(_added_lines(fp))
    if not removed or len(removed) != len(added):
        return False
    swapped = False
    for r, a in zip(removed, added):
        if r == a:
            continue
        if any(q in r or q in a for q in ("'", '"', "#", "\\")):
            return False  # strings and comments could hide the operators
        pr, pa = _CMP_SPLIT.split(r), _CMP_SPLIT.split(a)
        if len(pr) != len(pa):
            return False
        for i, (x, y) in enumerate(zip(pr, pa)):
            if i % 2 == 0:
                if x.strip() != y.strip():
                    return False
                continue
            x, y = " ".join(x.split()), " ".join(y.split())
            if x == y:
                continue
            if (x, y) not in _SWAPS:
                return False
            right = _OPERAND.match(pr[i + 1].lstrip())
            right_ok = right and _singleton_text(right.group()) and pr[i + 1].lstrip()[right.end():right.end() + 1] not in ("(", "[")
            left = re.search(r"([A-Za-z_][\w.]*)\s*$", pr[i - 1])
            left_ok = left and _singleton_text(left.group(1))
            if not (right_ok or left_ok):
                return False
            swapped = True
    return swapped


def _modname(path):
    parts = list(Path(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _run_module(path, text):
    """Execute `text` as module `path`, replacing it in sys.modules (child process only)."""
    import types as _t
    name = _modname(path)
    mod = _t.ModuleType(name)
    mod.__file__ = str(SRC / path)
    if path.endswith("__init__.py"):
        mod.__path__ = [str((SRC / path).parent)]
        mod.__package__ = name
    else:
        mod.__package__ = name.rpartition(".")[0]
    sys.modules[name] = mod
    exec(compile(text, str(SRC / path), "exec", dont_inherit=True), mod.__dict__)
    pkg, _, leaf = name.rpartition(".")
    if pkg in sys.modules:
        setattr(sys.modules[pkg], leaf, mod)


_ALL_LOADED = [False]


def _load_pristine(paths, every=False):
    """Import the pristine modules (and so their dependencies) into this process, once.
    every=True: all of sqlglot (failures ignored), for the state check's zygote."""
    import importlib
    import io
    import contextlib
    if every and _ALL_LOADED[0]:
        return
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    sys.dont_write_bytecode = True
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        if every:
            for m in _all_module_names():
                try:
                    importlib.import_module(m)
                except Exception:
                    pass
            _ALL_LOADED[0] = True
            return
        for p in paths:
            importlib.import_module(_modname(p))


def _child_runs(order, texts):
    """Fork; in the child run every module in `order` with texts[path] (or its pristine text)."""
    import signal
    sys.stdout.flush()
    pid = os.fork()
    if pid == 0:
        code = 3
        try:
            fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            signal.alarm(10)
            for p in order:
                _run_module(p, texts.get(p) if p in texts else (SRC / p).read_text())
            code = 0
        except BaseException:
            code = 3
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status) == 0


def _all_module_names():
    out = []
    for p in sorted((SRC / "sqlglot").rglob("*.py")):
        if p.name == "__main__.py":
            continue
        parts = list(p.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out.append(".".join(parts))
    return out


def _state_items(failed, only=None):
    """Deterministic text of every value sqlglot computed while being imported:
    module and class attributes, function defaults and closures, recursively.
    Functions and classes count by name (their code is covered by coverage)."""
    import enum
    on_path = set()
    skip_cls = {"__dict__", "__weakref__", "__doc__", "__module__", "__firstlineno__",
                "__static_attributes__", "__qualname__"}

    def walk(x, depth):
        t = type(x)
        if x is None or t in (bool, int, float, str, bytes, complex):
            return t.__name__ + ":" + repr(x)
        if isinstance(x, enum.Enum):
            return "enum:" + t.__qualname__ + "." + x.name
        if isinstance(x, (types.ModuleType, type)):
            return "ref:" + getattr(x, "__qualname__", getattr(x, "__name__", "?"))
        if id(x) in on_path:
            return "cycle"
        if depth > 80:
            return "deep:" + t.__qualname__
        on_path.add(id(x))
        try:
            return inner(x, t, depth + 1)
        finally:
            on_path.discard(id(x))

    def inner(x, t, depth):
        if isinstance(x, types.FunctionType):
            out = "fn:" + x.__qualname__
            if (x.__module__ or "").startswith("sqlglot"):
                out += "(" + walk(x.__defaults__, depth) + walk(x.__kwdefaults__, depth)
                if x.__closure__:
                    for cell in x.__closure__:
                        try:
                            out += walk(cell.cell_contents, depth)
                        except ValueError:
                            out += "empty"
                out += ")"
            return out
        if isinstance(x, (types.BuiltinFunctionType, types.MethodType, property, staticmethod, classmethod)):
            inner_fn = getattr(x, "__func__", None) or getattr(x, "fget", None)
            return "callable:" + getattr(x, "__qualname__", t.__qualname__) + (
                "(" + walk(inner_fn, depth) + ")" if isinstance(inner_fn, types.FunctionType) else "")
        if t in (list, tuple):
            items = [walk(v, depth) for v in x]
            if items and all(isinstance(v, type) for v in x):
                # classes hash by address, so a tuple built from a set of classes
                # (an isinstance() tuple) has no stable order: compare as a set
                items.sort()
            return t.__name__ + "[" + ",".join(items) + "]"
        if isinstance(x, (set, frozenset)):
            return "set{" + ",".join(sorted(walk(v, depth) for v in x)) + "}"
        if isinstance(x, dict):
            # dict order can follow the hash seed (built from sets): compare as a mapping
            return "{" + ",".join(sorted(walk(k, depth) + ":" + walk(v, depth) for k, v in x.items())) + "}"
        d = getattr(x, "__dict__", None)
        if not isinstance(d, dict):
            d = {}
        for klass in t.__mro__:
            for s in getattr(klass, "__slots__", ()) or ():
                if isinstance(s, str) and s not in d and hasattr(x, s):
                    d[s] = getattr(x, s)
        if d:
            return "obj:" + t.__qualname__ + "(" + walk(d, depth) + ")"
        r = repr(x)
        return "obj:" + t.__qualname__ + ("" if " at 0x" in r else ":" + r)

    import hashlib

    def h(text):
        return hashlib.blake2b(text.encode(), digest_size=10).hexdigest()

    def kind(v):
        # "d": plain data a recorder could stand in for; "f": anything callable or a descriptor
        if isinstance(v, (types.FunctionType, types.BuiltinFunctionType, types.MethodType,
                          staticmethod, classmethod, property, type, types.ModuleType)):
            return ":f"
        if hasattr(type(v), "__get__") or hasattr(type(v), "__set__"):
            return ":f"
        return ":d"

    out = {"FAILED": ",".join(sorted(failed))}
    for name in sorted(n for n in sys.modules if (n == "sqlglot" or n.startswith("sqlglot."))
                       and (only is None or n in only)):
        mod = sys.modules[name]
        f = getattr(mod, "__file__", None) or ""
        i = f.rfind("/sqlglot/")
        rel = f[i + 1:] if i >= 0 else name
        stack = []
        for k, v in sorted(vars(mod).items()):
            if k.startswith("__"):
                continue
            if isinstance(v, type) and v.__module__ == name and v.__qualname__ == k:
                stack.append(v)
            else:
                out[f"G|{rel}|{k}"] = h(walk(v, 0)) + kind(v)
        while stack:
            c = stack.pop()
            for ck, cv in sorted(vars(c).items()):
                if ck in skip_cls:
                    continue
                if isinstance(cv, type) and cv.__module__ == name and cv.__qualname__ == f"{c.__qualname__}.{ck}":
                    stack.append(cv)  # nested class: its own attributes
                    continue
                key = f"{rel}::{c.__qualname__}.{ck}"
                if type(cv) is dict and cv and all(" at 0x" not in repr(x) for x in cv):
                    # per-key hashes, so one changed entry selects only who looked it up
                    parts = []
                    for dk, dv in cv.items():
                        hk = h(walk(dv, 1))
                        out[f"{key}[{repr(dk)}]"] = hk + ":d"
                        parts.append(repr(dk) + "=" + hk)
                    out[key] = h("dict:" + ",".join(sorted(parts))) + ":d"
                else:
                    out[key] = h(walk(cv, 0)) + kind(cv)
    return out


def _stale_refs(fresh):
    """After re-importing the modules in `fresh` on top of a loaded package: does any
    other sqlglot module still hold an object (class or function) from their old
    versions? That module depends on them in a way the static import graph missed."""
    def current(obj):
        mod = sys.modules.get(getattr(obj, "__module__", None) or "")
        cur = mod
        for part in getattr(obj, "__qualname__", "").split("."):
            if part == "<locals>" or cur is None:
                return None
            cur = getattr(cur, part, None)
        return cur

    def check(v):
        if isinstance(v, (type, types.FunctionType)) and getattr(v, "__module__", None) in fresh:
            return current(v) is not v
        return False

    for name, mod in list(sys.modules.items()):
        if mod is None or name in fresh or not (name == "sqlglot" or name.startswith("sqlglot.")):
            continue
        values = list(vars(mod).values())
        for v in list(values):
            if isinstance(v, type) and v.__module__ == name:
                values.extend(vars(v).values())
        for v in values:
            items = list(v.values()) + list(v.keys()) if isinstance(v, dict) else \
                list(v) if isinstance(v, (list, tuple, set, frozenset)) else []
            if check(v) or any(check(x) for x in items):
                return True
    return False


def _state_in_child(texts, only=None):
    """Fork; the child imports all of sqlglot from scratch (patched files from `texts`)
    and sends back a digest of the import-time state. None if the child failed."""
    import importlib
    import importlib.abc
    import importlib.util
    import io
    import contextlib
    import signal
    sys.stdout.flush()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        code = 3
        try:
            os.close(r)
            fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            signal.alarm(60)
            for n in [n for n in sys.modules if n == "sqlglot" or n.startswith("sqlglot.")]:
                if only is None or n in only:
                    del sys.modules[n]
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))
            sys.dont_write_bytecode = True

            by_name = {}

            class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
                def find_spec(self, name, path, target=None):
                    rel = name.replace(".", "/")
                    for cand in (rel + ".py", rel + "/__init__.py"):
                        if cand in texts:
                            by_name[name] = cand
                            sub = [str(SRC / rel)] if cand.endswith("__init__.py") else None
                            return importlib.util.spec_from_file_location(
                                name, str(SRC / cand), loader=self, submodule_search_locations=sub)
                    return None

                def create_module(self, spec):
                    return None

                def exec_module(self, module):
                    rel = by_name[module.__name__]
                    exec(compile(texts[rel], module.__spec__.origin, "exec", dont_inherit=True), module.__dict__)

            if texts:
                sys.meta_path.insert(0, Finder())
            failed = []
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                for m in _all_module_names():
                    if only is not None and m not in only:
                        continue
                    try:
                        importlib.import_module(m)
                    except Exception as error:
                        failed.append(f"{m}:{type(error).__name__}")
            if only is not None and _stale_refs(only):
                items = {"STALE": "1"}
            else:
                items = _state_items(failed, only)
            payload = json.dumps(items, separators=(",", ":")).encode()
            view = memoryview(payload)
            while view:
                n = os.write(w, view)
                view = view[n:]
            code = 0
        except BaseException:
            code = 3
        finally:
            os._exit(code)
    os.close(w)
    data = b""
    while True:
        chunk = os.read(r, 1 << 16)
        if not chunk:
            break
        data += chunk
    os.close(r)
    _, status = os.waitpid(pid, 0)
    if os.waitstatus_to_exitcode(status) != 0 or not data:
        return None
    return json.loads(data.decode())


def _import_differential(patched, importers):
    """False only when the patched modules raise and the same run with pristine text does not."""
    paths = sorted(patched)
    _load_pristine(paths)
    order = list(paths)
    for p in paths:
        imps = [i for i in importers.get(p, ()) if i.startswith("sqlglot/") and _modname(i) in sys.modules]
        if len(imps) > IMPORT_CHECK_FANOUT:
            # a core module: re-running dozens of importers is slow and full of
            # re-execution artefacts; edits to its class bodies already reach
            # nearly every test through the class-users map
            continue
        for imp in imps:
            if imp not in order:
                order.append(imp)
    if _child_runs(order, patched):
        return True
    return not _child_runs(order, {})


class Selector:
    def __init__(self, tests):
        self.tests = tests
        self.all = [t["number"] for t in tests]
        self.all_bits = (1 << len(tests)) - 1
        self.by_file = {}
        self.by_id = {}
        for t in tests:
            self.by_file.setdefault(t["file"], []).append(t["number"])
            self.by_id[t["id"]] = t["number"]
        self.maps = Maps(len(tests))
        self.dur = {t["number"]: int(t["seconds"] * 1000 + 0.5) for t in tests}
        self._test_src = {}

    # -- helpers -----------------------------------------------------------

    def bits_of(self, numbers):
        out = 0
        for n in numbers:
            out |= 1 << n
        return out

    def file_tests(self, f):
        return self.bits_of(self.by_file.get(f, ()))

    def _test_file_text(self, f):
        if f not in self._test_src:
            try:
                self._test_src[f] = (SRC / f).read_text()
            except OSError:
                self._test_src[f] = ""
        return self._test_src[f]

    def mentioned_by_tests(self, needles):
        out = 0
        for f in self.by_file:
            text = self._test_file_text(f)
            if any(nd in text for nd in needles):
                out |= self.file_tests(f)
        return out

    def import_sensitive(self):
        out = 0
        for key in IMPORT_SENSITIVE:
            if key.endswith(".py"):
                out |= self.file_tests(key)
            elif key in self.by_id:
                out |= 1 << self.by_id[key]
        return out

    def importers_closure(self, path, limit_prefix=None):
        seen = {path}
        todo = [path]
        while todo:
            cur = todo.pop()
            for imp in self.maps.imports.get(cur, ()):
                if imp not in seen and (limit_prefix is None or imp.startswith(limit_prefix)):
                    seen.add(imp)
                    todo.append(imp)
        return seen

    def family(self, path):
        """Modules whose code shares state with an import-time change in `path`."""
        parts = path.split("/")
        mods = {path}
        if len(parts) == 3 and parts[0] == "sqlglot" and parts[1] in FAMILY_DIRS:
            stem = parts[2]
            for d in FAMILY_DIRS:
                cand = f"sqlglot/{d}/{stem}"
                if self.maps.struct(cand) is not None:
                    mods.add(cand)
        # dialect subclasses and helpers living in other files import their parent
        for m in list(mods):
            for imp in self.maps.imports.get(m, ()):
                ip = imp.split("/")
                if len(ip) == 3 and ip[0] == "sqlglot" and ip[1] in FAMILY_DIRS:
                    mods.add(imp)
        return mods

    def family_union(self, path):
        out = 0
        for m in self.family(path):
            out |= self.maps.file_union(m)
        return out

    BINDING_RE = re.compile(r"^\s*(?:(?:async\s+)?def\s+(\w+)|class\s+(\w+)|(\w+)\s*(?::[^=]*)?=(?!=))")

    def _source_files(self):
        if not hasattr(self, "_sources"):
            self._sources = []
            for top in ("sqlglot", "tests"):
                for p in sorted((SRC / top).rglob("*.py")):
                    try:
                        self._sources.append((str(p.relative_to(SRC)), p.read_text()))
                    except OSError:
                        pass
        return self._sources

    def _source_paths(self):
        if not hasattr(self, "_source_set"):
            self._source_set = {rel for rel, _ in self._source_files()}
        return self._source_set

    def _defined_elsewhere(self, name, path):
        """Is `name` bound as a class attribute in some other sqlglot class (a base to fall back to)?"""
        pat = re.compile(r"^[ \t]+%s\s*(?::[^=\n]*)?=" % re.escape(name), re.M)
        for rel, text in self._source_files():
            if rel != path and rel.startswith("sqlglot/") and pat.search(text):
                return True
        return False

    def _patched_texts(self, files):
        """New text of every .py file the patch modifies or adds (None if unknown)."""
        out = {}
        for fp in files:
            path = fp.path or ""
            if not path.endswith(".py") or fp.binary:
                continue
            if fp.old is None and fp.new:
                out[fp.new] = "\n".join(_added_lines(fp))
            elif fp.old and fp.new and fp.old == fp.new:
                try:
                    old_text = (SRC / fp.old).read_text()
                except OSError:
                    continue
                new_text = apply_hunks(old_text, fp.hunks)
                if new_text is not None:
                    out[fp.old] = new_text
        return out

    # -- import check ------------------------------------------------------
    # Code that raises while a module is imported (a class body popping a missing
    # key, a bad constant) fails every test that loads the module, which coverage
    # cannot see. Run the patched modules (and the sqlglot modules that import them)
    # in a forked child on top of the pristine package, and run the same modules with
    # their pristine text in another child: only "pristine runs, patched raises"
    # counts, so artefacts of re-running a module do not.

    _pkg_loaded = None

    def _imports_cleanly(self):
        if getattr(self, "_import_verdict", None) is not None:
            return self._import_verdict
        patched = {p: t for p, t in getattr(self, "_patched", {}).items()
                   if p.startswith("sqlglot/") and (SRC / p).exists()}
        verdict = True
        if patched and hasattr(os, "fork"):
            try:
                verdict = _import_differential(patched, self.maps.imports)
            except Exception:
                verdict = True  # the check itself broke: no signal either way
        self._import_verdict = verdict
        return verdict

    def _changed_state(self):
        """Keys of import-time values (module globals, class attributes) whose value
        differs once the patched package is imported; None when that is unknown."""
        if getattr(self, "_state_done", False):
            return self._state_changed
        changed = None
        patched = {p: t for p, t in getattr(self, "_patched", {}).items()
                   if p.startswith("sqlglot/") and (SRC / p).exists()}
        if patched and hasattr(os, "fork"):
            try:
                closure = set(patched)
                todo = list(patched)
                while todo:
                    for imp in self.maps.imports.get(todo.pop(), ()):
                        if imp.startswith("sqlglot/") and imp not in closure:
                            closure.add(imp)
                            todo.append(imp)
                if len(closure) <= ZYGOTE_MAX_MODULES:
                    # re-import only the patched modules and their importers on top of
                    # the loaded package, once with pristine and once with patched text
                    only = {_modname(p) for p in closure}
                    _load_pristine(sorted(SRC.joinpath("sqlglot").rglob("*.py")), every=True)
                    base = _state_in_child({}, only)
                    new = _state_in_child(patched, only) if base else None
                    if base and new and "STALE" not in base and "STALE" not in new:
                        changed = {k for k in set(base) | set(new) if base.get(k) != new.get(k)}
                        self._state_base = base
                        self._state_new = new
                if changed is None:
                    if getattr(Selector, "_pristine_state", None) is None:
                        Selector._pristine_state = _state_in_child({}) or {}
                    base = Selector._pristine_state
                    new = _state_in_child(patched) if base else None
                    if new is not None:
                        changed = {k for k in set(base) | set(new) if base.get(k) != new.get(k)}
                        self._state_base = base
                        self._state_new = new
            except Exception:
                changed = None
        self._state_done = True
        self._state_changed = changed
        return changed

    def _import_state_unchanged(self):
        """Does importing all of sqlglot with this commit's patch leave every
        import-time value (module/class attributes, defaults, closures) as it was?"""
        return self._changed_state() == set()

    def _est_cost(self, bits):
        """Busiest worker + worker fee of the packing `pack` would choose, in ms."""
        sel = []
        n = 0
        while bits:
            if bits & 1:
                sel.append(n)
            bits >>= 1
            n += 1
        workers = pack(sel, self.dur)
        if not workers:
            return 0
        return max(sum(self.dur[t] for t in w) for w in workers) + WORKER_COST_MS * len(workers)

    def _name_readers(self, name):
        """Tests that run a line naming `name` (a changed module-level value), and every
        test of a test file that names it."""
        pat = re.compile(r"\b%s\b" % re.escape(name))
        out = 0
        for rel, text in self._source_files():
            if name not in text:
                continue
            if rel.startswith("tests/"):
                out |= self.file_tests(rel)
                continue
            for i, line in enumerate(text.split("\n"), 1):
                if name in line and pat.search(line):
                    out |= self.maps.line(rel, i)
        return out

    def _new_value_readers(self, key, new):
        """A data value that did not exist at the pinned commit (a new class setting or
        module constant). Pinned code that reads that name now gets it: through a class
        attribute the readers of every same-named attribute of any class (a subclass
        setting shadows its parent's), and the lines naming it (module globals, and
        builtins or star-imported names it may shadow). New code that reads it is in the
        patch, whose touched lines select on their own. None for anything callable."""
        if new is None or not new.endswith(":d"):
            return None
        if key.startswith("G|"):
            return self._name_readers(key.rsplit("|", 1)[1])
        if "::" not in key:
            return None
        if "[" in key:
            # an entry of a dict: covered by the dict itself when the whole dict is new
            whole = key[:key.index("[")]
            return 0 if whole not in self._state_base and whole in self._state_new else None
        name = key.rsplit(".", 1)[1]
        out = self._name_readers(name)
        maps = self.maps
        maps.attr_readers("__unreliable__")  # loads the index
        tail = "." + name
        for k in maps._attrs:
            if k.endswith(tail):
                out |= maps.attr_readers(k) or 0
        return out

    def _method_callers(self, key, new):
        """A method defined in a class body was removed, redefined or re-decorated
        (state key "sqlglot/x.py::Cls.meth", old value a function). Only a test that
        called it at the pinned commit (ran a line of its body) can see the difference:
        generator dispatch (`<key>_sql` found through dir()) calls it, and every
        import-time table derived from it is a state key of its own in the same diff.
        None when the old value is not a multi-line def of that class in that file."""
        if new is not None and not new.endswith(":f"):
            return None
        if key.startswith("G|"):
            # a module-level function (its defaults or closure changed, or it was
            # removed), seen in the module that defines it and in every module that
            # imported it by name: the callers of the def, wherever it is defined
            _g, rel0, name = key.split("|", 2)
            homes = [rel0] + sorted({k.split("|", 2)[1] for k in (self._state_changed or ())
                                     if k.startswith("G|") and k.rsplit("|", 1)[1] == name} - {rel0})
            for rel in homes:
                st = self.maps.struct(rel)
                if st is None:
                    continue
                for kind, qual, start, end, body in st["regions"]:
                    if kind == "d" and qual == name:
                        if body <= start or body > end:
                            return None
                        return self.maps.lines(rel, body, end)
            return None
        if "::" not in key:
            return None
        rel, qual = key.split("::", 1)
        st = self.maps.struct(rel)
        if st is None:
            return None
        for kind, name, start, end, body in st["regions"]:
            if kind == "d" and name == qual:
                if body <= start or body > end:
                    return None  # one-line def: its body shares the def line
                return self.maps.lines(rel, body, end)
        return None

    def _readers_of_changes(self):
        """Tests that read an import-time value this commit changes, or None when a
        changed value cannot be mapped to its readers (then the caller widens)."""
        changed = self._changed_state()
        if changed is None or "FAILED" in changed:
            return None
        base = self._state_base
        new_kind = {}
        out = 0
        maps = self.maps
        # nothing in sqlglot or its tests reads class annotations at run time; what a
        # decorator derives from them (dataclass fields, __init__) is a state key of its
        # own in the same diff
        changed = {k for k in changed if k.startswith("G|") or ".__annotations__" not in k}
        keyed = {}  # dict attribute -> changed entries
        for key in changed:
            if key.endswith("]") and "[" in key and not key.startswith("G|"):
                attr = key[:key.index("[")]
                if attr in base:
                    keyed.setdefault(attr, []).append(key)
        for key in sorted(changed):
            if key in keyed or (key.endswith("]") and key[:key.find("[")] in keyed):
                continue
            old = base.get(key)
            if old is None:
                readers = self._new_value_readers(key, self._state_new.get(key))
                if readers is None:
                    return None  # a new attribute or global: who reads it is unknown
                out |= readers
                continue
            if not old.endswith(":d"):
                callers = self._method_callers(key, self._state_new.get(key))
                if callers is None:
                    return None  # a function, descriptor or class: defaults/closures changed
                out |= callers
                continue
            if key.startswith("G|"):
                out |= self._name_readers(key.rsplit("|", 1)[1])
                continue
            readers = maps.attr_readers(key)
            if readers is None:
                return None  # no recorder stood in for it
            out |= readers
        for attr, entries in keyed.items():
            if maps.attr_readers(attr + "{KEYED}") is None:
                # the recorder did not watch this dict key by key: all its readers
                readers = maps.attr_readers(attr)
                if readers is None:
                    return None
                out |= readers
                continue
            # entries changed, added or removed: who looked those keys up (present or
            # missing), plus whoever used the dict as a whole (iterated, copied, merged)
            out |= maps.attr_readers(attr + "{ALL}") or 0
            for e in entries:
                out |= maps.attr_readers(e) or 0
        if changed:
            out |= maps.attr_readers("__unreliable__") or 0
        return out

    def check_removed_names(self, fp, path, old_text, reg, regions):
        """A class- or module-level name that disappears breaks every module that
        refers to it (an enum member, a constant, an imported helper)."""
        added = set()
        for c in _added_lines(fp):
            m = self.BINDING_RE.match(c)
            if m:
                added.add(next(g for g in m.groups() if g))
        removed = []
        for os_, ol, _ns, _nl, body in fp.hunks:
            o = os_
            for hl in body:
                if hl[:1] == "-":
                    m = self.BINDING_RE.match(hl[1:])
                    r = reg[o - 1] if 0 < o <= len(reg) else -1
                    at_top = r < 0 or regions[r][0] == "c" or regions[r][2] == o
                    if m and at_top:
                        name = next(g for g in m.groups() if g)
                        if name not in added and not name.startswith("__"):
                            removed.append((name, r))
                if hl[:1] in " -":
                    o += 1
        class_names = {rg[1] for rg in regions if rg[0] == "c"}
        for name, r in removed:
            owner = None  # qualname of the class the removed name belongs to
            if r >= 0:
                if regions[r][0] == "c":
                    owner = regions[r][1]
                elif "." in regions[r][1] and regions[r][1].rsplit(".", 1)[0] in class_names:
                    owner = regions[r][1].rsplit(".", 1)[0]  # a method
            if owner:
                if not (name.isupper() and len(name) > 1):
                    continue  # an inherited attribute or method falls back to the base class
                if not _is_enum_class(owner.rsplit(".", 1)[-1]) and self._defined_elsewhere(name, path):
                    continue  # an overridden class setting falls back to the base value
            pat = re.compile(r"\b%s\b" % re.escape(name))
            # search the tree as it is after the whole patch: a commit that removes
            # a helper together with its callers leaves nothing that refers to it
            patched = getattr(self, "_patched", {})
            for rel, text in self._source_files():
                text = patched.get(rel, text)
                if pat.search(text):
                    if rel == path:
                        raise Unknown(f"removed name {name} still used in its module")
                    raise Unknown(f"removed name {name} used in {rel}")
            for rel, text in patched.items():
                if rel not in self._source_paths() and pat.search(text):
                    raise Unknown(f"removed name {name} used in new file {rel}")

    # -- per file kinds ----------------------------------------------------

    def python_file(self, fp, is_test):
        path = fp.old
        st = self.maps.struct(path)
        try:
            old_text = (SRC / path).read_text()
        except OSError:
            raise Unknown(path)
        new_text = apply_hunks(old_text, fp.hunks)
        if new_text is None:
            raise Unknown("hunks do not apply")
        if all((not c.strip()) or c.strip().startswith("#") for c in _changed_lines(fp)):
            if _ast_equal(old_text, new_text):
                return 0
        elif _is_eq_rewrite(fp):
            return 0
        if _docstring_only(fp, old_text, new_text):
            return 0
        try:
            ast.parse(new_text)
        except (SyntaxError, ValueError):
            if is_test:
                out = 0
                for f in self.importers_closure(path, "tests/"):
                    out |= self.file_tests(f)
                return out
            raise Unknown("syntax error")
        if st is None:
            raise Unknown("no structure")

        maps = self.maps
        n = st["n"]
        reg, stmt, regions = st["reg"], st["stmt"], st["regions"]
        touched, removed = old_lines_touched(fp, n)
        # header lines (decorators, def/class signature): they run when the module is
        # imported, but a change in the body below them does not re-run them
        header = set()
        for rg in regions:
            header.update(range(rg[2], rg[4]))
        added_def = any(c.lstrip().startswith("@") for c in _added_lines(fp))
        if not added_def and any(c.lstrip().startswith(("def ", "async def ", "class "))
                                 for c in _added_lines(fp)):
            # a def/class line was added: a new definition unless every def and class
            # of the patched file already exists (same qualified name) at the pinned
            # commit, i.e. only an existing signature or class line was rewritten
            old_names, new_names = _def_names(old_text), _def_names(new_text)
            added_def = old_names is None or new_names is None or not new_names <= old_names
        if not is_test:
            self.check_removed_names(fp, path, old_text, reg, regions)
        out = 0
        wide = 0  # class-user widening, replaced by attribute readers when they are known
        import_time = added_def
        runs_on_import = added_def
        strict_import = added_def  # import-time reasons other than "a function body ran at import"
        at_import = set(maps.import_lines(path))
        class_names = {rg[1] for rg in regions if rg[0] == "c"}
        classes_hit = set()

        headers = []

        def same_header(qual):
            if not headers:
                headers.append((_def_headers(old_text), _def_headers(new_text)))
            old_h, new_h = headers[0]
            if old_h is None or new_h is None:
                return False
            return qual in old_h and old_h.get(qual) == new_h.get(qual)

        def enclosing_class(r):
            name = regions[r][1]
            if regions[r][0] == "c":
                return name
            while "." in name:
                name = name.rsplit(".", 1)[0]
                if name in class_names:
                    return name
            return None

        for ln in sorted(touched):
            out |= maps.line(path, ln)
            s = stmt[ln - 1]
            if s in header:
                # the innermost statement is the def/class itself (a blank or comment
                # line in a body): no multi-line statement to widen to
                s = 0
            if s and s < ln:
                # a multi-line statement: every line from its start up to the change
                # (e.g. an `and` turned `or` in a later line of an `if (...)` header)
                out |= maps.lines(path, s, ln)
            ran_at_import = any(x in at_import and x not in header
                                for x in range(s if s and s < ln else ln, ln + 1))
            r = reg[ln - 1]
            if ran_at_import:
                runs_on_import = True
            if r < 0:  # module level
                import_time = True
                strict_import = True
                runs_on_import = True
                continue
            cls_name = enclosing_class(r)
            if cls_name:
                classes_hit.add(cls_name)
            kind, name, start, end, body = regions[r]
            if kind == "c" or ln < body:
                runs_on_import = True  # class body, decorator or signature
            if kind == "c":
                # class-body state (arg_types, dialect settings, tables) reaches the
                # tests that use the class or a subclass, wherever the reading code is
                users = maps.class_users(path, name)
                if users and not is_test:
                    wide |= users
                else:
                    # never instantiated (enums, mixins, namespaces): widen to the
                    # users of every class in the file, plus the family fallback
                    for rg in regions:
                        if rg[0] == "c":
                            wide |= maps.class_users(path, rg[1])
                    import_time = True
                    strict_import = True
            elif ln == body - 1 and ln not in removed and ln + 1 in touched:
                # unchanged last header line, touched only as the neighbour of an
                # insertion at the top of the body: the body lines carry the change
                pass
            elif ln < body and kind == "d" and same_header(name):
                # only annotations changed (unevaluated strings): the body lines and
                # coverage of this line carry whatever else changed
                pass
            elif ln < body:  # decorator or signature: affects every call
                out |= maps.lines(path, body, end)
                # decorators and defaults are evaluated at import time too
                import_time = True
                strict_import = True
            elif ran_at_import:
                # a function body that runs while modules are imported or a dialect
                # is first set up (table builders, dispatch caches): its effect
                # outlives any single test
                import_time = True
            if is_test and kind == "d":
                # the test function itself, even if coverage never reached the line
                cls = name.rsplit(".", 1)
                if len(cls) == 2:
                    tid = f"{path}::{cls[0].replace('.', '::')}::{cls[1]}"
                else:
                    tid = f"{path}::{name}"
                if tid in self.by_id:
                    out |= 1 << self.by_id[tid]
        if runs_on_import and not is_test and not self._imports_cleanly():
            raise Unknown("patched module raises when imported")
        broad = out | wide
        if import_time and not is_test and not path.endswith("__init__.py"):
            for cls_name in classes_hit:
                broad |= maps.class_users(path, cls_name)
            broad |= self.family_union(path) | self.import_sensitive()
        # the state check (a full import of the patched package in a child) costs
        # about a CPU second: run it only when the broad selection costs clearly more
        worth_checking = self._est_cost(broad) - self._est_cost(out) > STATE_CHECK_MIN_GAIN_MS
        if (wide or import_time) and not is_test and not added_def and worth_checking \
                and not path.endswith("__init__.py") \
                and not any("lambda" in c for c in _changed_lines(fp)):
            # import-time state: instead of every user of the class (and the module's
            # family), the tests that read the values the patch actually changes
            # (attribute-read map from a recorder pass over the pinned suite; module
            # globals by the lines that name them). Lambdas are left out: a one-line
            # lambda's body runs on its definition line, which coverage may not credit.
            readers = self._readers_of_changes()
            if readers is not None:
                return out | readers
        out |= wide
        if import_time and not strict_import and not is_test and worth_checking \
                and not path.endswith("__init__.py") and self._import_state_unchanged():
            # only function bodies that ran during import changed, and importing
            # the patched package leaves every import-time value as it was: what
            # the edit can still change, it changes while tests run (coverage)
            import_time = False
        if import_time:
            # class-level state (arg_types, dialect settings, tables) reaches every
            # test that uses the class or a subclass, wherever the reading code lives
            for cls_name in classes_hit:
                out |= maps.class_users(path, cls_name)
            if is_test:
                for f in self.importers_closure(path, "tests/"):
                    out |= self.file_tests(f)
                out |= self.family_union(path)
            else:
                if path.endswith("__init__.py"):
                    raise Unknown("package init at import time")
                out |= self.family_union(path)
                out |= self.import_sensitive()
        return out

    def fixture_file(self, fp):
        path = fp.path
        opened = self.maps.fixtures.get(path)
        if opened:
            return self.bits_of(x for x in opened if isinstance(x, int))
        rel = path[len("tests/fixtures/"):]
        return self.mentioned_by_tests((rel, os.path.basename(path)))

    def other_file(self, fp):
        path = fp.path
        if path.startswith("sqlglot/") or path.startswith("tests/"):
            raise Unknown(path)
        parts = path.split("/")
        needles = {os.path.basename(path)}
        if len(parts) > 1:
            needles.add(parts[0])
        return self.mentioned_by_tests(needles)

    # -- entry ---------------------------------------------------------------

    def select_bits(self, patch_text):
        files = parse_patch(patch_text)
        if not files:
            raise Unknown("empty patch")
        out = 0
        self._patched = self._patched_texts(files)
        self._import_verdict = None
        self._state_done = False
        self._state_changed = None
        for fp in files:
            if fp.binary:
                raise Unknown("binary")
            path = fp.path or ""
            if fp.old is None or fp.new is None or fp.old != fp.new:
                # added, deleted or renamed file
                if fp.old is None and path.endswith(".py") and path.startswith("sqlglot/"):
                    out |= self.family_union(path) | self.import_sensitive()
                    continue
                if fp.old is None and not path.startswith(("sqlglot/", "tests/")):
                    out |= self.other_file(fp)
                    continue
                raise Unknown("file added/removed/renamed")
            if path.endswith(".py") and path.startswith("sqlglot/"):
                out |= self.python_file(fp, is_test=False)
            elif path.endswith(".py") and path.startswith("tests/"):
                out |= self.python_file(fp, is_test=True)
            elif path.startswith("tests/fixtures/"):
                out |= self.fixture_file(fp)
            else:
                out |= self.other_file(fp)
        return out

    def select(self, patch_text):
        try:
            bits = self.select_bits(patch_text)
        except Unknown as why:
            log(f"  unknown ({why}): all tests")
            bits = self.all_bits
        out = []
        n = 0
        while bits:
            if bits & 1:
                out.append(n)
            bits >>= 1
            n += 1
        return set(out)


# ---------------------------------------------------------------- packing

def pack(selected, dur):
    """Split onto <= 8 workers minimising max load + 2 s per worker."""
    if not selected:
        return []
    items = sorted(selected, key=lambda n: (-dur[n], n))
    total = sum(dur[n] for n in items)
    longest = dur[items[0]]
    best = None
    for k in range(1, MAX_WORKERS + 1):
        lower = max(longest, -(-total // k)) + WORKER_COST_MS * k
        if best is not None and lower >= best[0]:
            if longest >= -(-total // k):
                break
            continue
        loads = [0] * k
        bins = [[] for _ in range(k)]
        for n in items:
            j = loads.index(min(loads))
            loads[j] += dur[n]
            bins[j].append(n)
        cost = max(loads) + WORKER_COST_MS * k
        if best is None or cost < best[0]:
            best = (cost, bins)
    return [b for b in best[1] if b]


def main():
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    dur = {t["number"]: int(t["seconds"] * 1000 + 0.5) for t in tests}
    selector = Selector(tests)
    out = []
    for patch in sorted((DATA / "commits").glob("*.patch")):
        cid = patch.stem
        try:
            text = patch.read_text(encoding="utf-8", errors="replace")
            selected = selector.select(text)
        except Exception as error:  # never crash: fall back to everything
            log(f"{cid}: error {error!r}, running all")
            selected = set(selector.all)
        workers = pack(selected, dur)
        log(f"{cid}: {len(selected)} tests on {len(workers)} workers")
        out.append(json.dumps({"commit": cid, "workers": workers}, separators=(",", ":")))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
