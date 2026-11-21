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
from pathlib import Path

DATA = Path(os.environ.get("BR_DATA", "/data"))
SRC = Path(os.environ.get("BR_SRC", "/app/src"))  # sqlglot at eb11aec
APP = Path(os.environ.get("BR_APP", Path(__file__).resolve().parent))
STRUCT = Path(os.environ.get("BR_STRUCT", APP / "struct"))
MAPS = APP / "maps"
MAX_WORKERS = 8
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
    """Old-file line numbers removed, plus both neighbours of every insertion."""
    out = set()
    for os_, ol, _ns, _nl, body in fp.hunks:
        o = os_ if ol else os_ + 1  # line number of the next old line
        prev_tag = None
        for hl in body:
            tag = hl[:1]
            if tag == "-":
                out.add(o)
                o += 1
            elif tag == "+":
                out.add(o - 1)
                out.add(o)
            else:
                o += 1
            prev_tag = tag
    return {x for x in out if 1 <= x <= n}


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


def _ast_equal(a, b):
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except (SyntaxError, ValueError):
        return False


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
        for name, r in removed:
            in_class = r >= 0 and regions[r][0] == "c"
            if in_class and not (name.isupper() and len(name) > 1):
                continue  # an inherited attribute or method falls back to the base class
            pat = re.compile(r"\b%s\b" % re.escape(name))
            for rel, text in self._source_files():
                if rel == path:
                    hits = len(pat.findall(text))
                    if hits > len(pat.findall("\n".join(c for c in _changed_lines(fp)))):
                        raise Unknown(f"removed name {name} still used in its module")
                elif pat.search(text):
                    raise Unknown(f"removed name {name} used in {rel}")

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
        touched = old_lines_touched(fp, n)
        added_def = any(c.lstrip().startswith(("def ", "async def ", "class ", "@")) for c in _added_lines(fp))
        if not is_test:
            self.check_removed_names(fp, path, old_text, reg, regions)
        out = 0
        import_time = added_def
        at_import = set(maps.import_lines(path))
        class_names = {rg[1] for rg in regions if rg[0] == "c"}
        classes_hit = set()

        def enclosing_class(r):
            name = regions[r][1]
            if regions[r][0] == "c":
                return name
            while "." in name:
                name = name.rsplit(".", 1)[0]
                if name in class_names:
                    return name
            return None

        for ln in touched:
            out |= maps.line(path, ln)
            s = stmt[ln - 1]
            if s and s < ln:
                # a multi-line statement: every line from its start up to the change
                # (e.g. an `and` turned `or` in a later line of an `if (...)` header)
                out |= maps.lines(path, s, ln)
            ran_at_import = any(x in at_import for x in range(s if s and s < ln else ln, ln + 1))
            r = reg[ln - 1]
            if r < 0:  # module level
                import_time = True
                continue
            cls_name = enclosing_class(r)
            if cls_name:
                classes_hit.add(cls_name)
            kind, name, start, end, body = regions[r]
            if kind == "c":
                # class-body state (arg_types, dialect settings, tables) reaches the
                # tests that use the class or a subclass, wherever the reading code is
                users = maps.class_users(path, name)
                if users and not is_test:
                    out |= users
                else:
                    # never instantiated (enums, mixins, namespaces): widen to the
                    # users of every class in the file, plus the family fallback
                    for rg in regions:
                        if rg[0] == "c":
                            out |= maps.class_users(path, rg[1])
                    import_time = True
            elif ln < body:  # decorator or signature: affects every call
                out |= maps.lines(path, body, end)
                # decorators and defaults are evaluated at import time too
                import_time = True
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
