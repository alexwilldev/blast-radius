"""Build-time structure maps for the pinned sqlglot tree (run in the Dockerfile).

For every .py file under sqlglot/ and tests/ writes <out>/<path>.json with:
  n:     number of lines
  reg:   per line (index 0 = line 1), innermost def/class region id, or -1
  stmt:  per line, first line of the innermost statement containing it, or 0
  regions: [kind, qualname, start, end, body_start]   kind: "d" def, "c" class
Also writes _imports.json: module path -> files that import it directly.
"""

import ast
import json
import os
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])


def module_path(mod):
    base = SRC / Path(*mod.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.exists():
            return str(cand.relative_to(SRC))
    return None


def struct(path):
    text = path.read_text()
    tree = ast.parse(text)
    n = text.count("\n") + 1
    reg = [-1] * n
    stmt = [0] * n
    regions = []
    imports = set()

    def visit(node, parent_name, parent_id):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                s, e = child.lineno, child.end_lineno or child.lineno
                for i in range(s - 1, min(e, n)):
                    stmt[i] = s
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([d.lineno for d in child.decorator_list] + [child.lineno])
                end = child.end_lineno or child.lineno
                name = f"{parent_name}.{child.name}" if parent_name else child.name
                rid = len(regions)
                regions.append(["c" if isinstance(child, ast.ClassDef) else "d", name, start, end,
                                child.body[0].lineno])
                for i in range(start - 1, min(end, n)):
                    reg[i] = rid
                # decorators belong to the def statement
                for i in range(start - 1, min(end, n)):
                    if stmt[i] < start:
                        stmt[i] = start
                visit(child, name, rid)
            else:
                if isinstance(child, ast.Import):
                    for a in child.names:
                        imports.add(a.name)
                elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                    imports.add(child.module)
                    for a in child.names:
                        imports.add(f"{child.module}.{a.name}")
                elif isinstance(child, ast.ImportFrom) and child.level:
                    pkg = str(path.relative_to(SRC).parent).replace("/", ".").split(".")
                    pkg = pkg[: len(pkg) - (child.level - 1)]
                    mod = ".".join(pkg + ([child.module] if child.module else []))
                    imports.add(mod)
                    for a in child.names:
                        imports.add(f"{mod}.{a.name}")
                visit(child, parent_name, parent_id)

    visit(tree, "", -1)
    return {"n": n, "reg": reg, "stmt": stmt, "regions": regions}, imports


def main():
    importers = {}
    count = 0
    for top in ("sqlglot", "tests"):
        for path in sorted((SRC / top).rglob("*.py")):
            rel = str(path.relative_to(SRC))
            try:
                data, imports = struct(path)
            except SyntaxError:
                continue
            dest = OUT / (rel + ".json")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(data, separators=(",", ":")))
            count += 1
            for mod in imports:
                mp = module_path(mod)
                if mp and mp != rel:
                    importers.setdefault(mp, set()).add(rel)
    (OUT / "_imports.json").write_text(
        json.dumps({k: sorted(v) for k, v in sorted(importers.items())}, separators=(",", ":")))
    print(f"structure for {count} files", file=sys.stderr)


if __name__ == "__main__":
    main()
