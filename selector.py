"""Blast radius selector.

For each patch in /data/commits, decide which sqlglot tests could fail and pack
them onto workers. Logs go to stderr; stdout carries one JSON line a commit.

v0: provably harmless edits (comment/blank-only Python changes whose AST is
unchanged) run nothing; fixture-only edits run the test files that load that
fixture; anything else runs every test, packed as tightly as possible.
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

DATA = Path(os.environ.get("BR_DATA", "/data"))
SRC = Path(os.environ.get("BR_SRC", "/app/src"))  # sqlglot at eb11aec
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


def _changed_lines(fp):
    for _os, _ol, _ns, _nl, body in fp.hunks:
        for hl in body:
            if hl[:1] in "+-":
                yield hl[1:]


def _ast_equal(a, b):
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except (SyntaxError, ValueError):
        return False


def python_is_harmless(fp):
    """Comment/blank-only edit to an existing .py file whose AST is unchanged."""
    if fp.old is None or fp.new is None or fp.old != fp.new or fp.binary:
        return False
    for content in _changed_lines(fp):
        s = content.strip()
        if s and not s.startswith("#"):
            return False
    src = SRC / fp.old
    try:
        old_text = src.read_text()
    except OSError:
        return False
    new_text = apply_hunks(old_text, fp.hunks)
    if new_text is None:
        return False
    return _ast_equal(old_text, new_text)


class Selector:
    def __init__(self, tests):
        self.tests = tests
        self.all = [t["number"] for t in tests]
        self.by_file = {}
        for t in tests:
            self.by_file.setdefault(t["file"], []).append(t["number"])
        self._test_src = {}

    def _test_file_text(self, f):
        if f not in self._test_src:
            try:
                self._test_src[f] = (SRC / f).read_text()
            except OSError:
                self._test_src[f] = None
        return self._test_src[f]

    def fixture_tests(self, path):
        """Test files whose source names this fixture (by relative path or basename)."""
        rel = path[len("tests/fixtures/"):]
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        out = set()
        for f in self.by_file:
            text = self._test_file_text(f)
            if text is None:
                return None
            if rel in text or base in text or (len(stem) > 3 and f'"{stem}' in text):
                out.update(self.by_file[f])
        if not out:
            return None
        return out

    def select(self, patch_text):
        files = parse_patch(patch_text)
        if not files:
            return set(self.all)
        chosen = set()
        for fp in files:
            path = fp.path or ""
            if path.endswith(".py") and python_is_harmless(fp):
                continue
            if (path.startswith("tests/fixtures/") and fp.old and fp.new
                    and fp.old == fp.new and not fp.binary):
                got = self.fixture_tests(path)
                if got is not None:
                    chosen |= got
                    continue
            return set(self.all)
        return chosen


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
