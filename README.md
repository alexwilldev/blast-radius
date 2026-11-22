# Blast Radius entry

## How this entry selects tests

Everything is derived from sqlglot at the pinned commit `eb11aec`. No table maps patches, or kinds of edit, to tests.

- `maps/coverage.idx` + `maps/coverage.sets`: per-test line coverage of `sqlglot/` and `tests/` (pytest-cov `--cov-context=test`). The suite's `ProcessPoolExecutor`s were run in-process during the coverage run, with pickling kept, so work done in pool workers counts toward the test that submitted it. Identical test sets are stored once as bitsets.
- `maps/fixtures.json`: which tests open which fixture files (recorded with an `open` audit hook). A file read during a test's setup phase (`setUpClass`) counts for every test in that class.
- `maps/import_lines.z`: lines executed just by importing every module and creating each dialect's tokenizer, parser and generator once. These run once per process (dialect tables, dispatch caches), so coverage can't credit them to the tests that depend on them.
- `maps/usage.idx` + `maps/usage.sets`: which tests instantiate each sqlglot class (or look it up as a dialect), counting every sqlglot class in its MRO. Built in a separate instrumented run of the suite.
- `build_struct.py` (runs in the Dockerfile): def/class/statement spans and the static import graph of the pinned source.
- `selector.py`, for each patch:
  - A changed line selects the tests that execute it, or any earlier line of the same multi-line statement.
  - Class-body edits (attributes, `arg_types`, dialect settings and tables) select the tests that use that class or a subclass. Classes that are never instantiated (enums, mixins) widen to the users of every class in the file.
  - Removing a module-level name, or an upper-case class member such as an enum value, that code still refers to selects every test. References are searched in the tree as it is after the whole patch, so a helper removed together with its callers does not count. A method, or an upper-case setting of a non-Enum class that another class also defines, falls back to the base class and is left to coverage.
  - Other import-time code (module level, signatures, decorators, new definitions, and function bodies in `import_lines.z`) selects every test that executes code in the module's dialect family or its importers.
  - A `def`/`class` header line is not itself an import-time trigger when only the body changed: a blank or comment line in a body is not widened to the enclosing `def`, and an insertion right below an unchanged signature counts as a body edit.
  - Behaviour-preserving rewrites select nothing: if every changed line is unchanged except for `is`/`is not` turned into `==`/`!=` (or back) against `None` or an Enum member (Enum found by scanning the sqlglot source), with no strings or comments on the line, the edit runs no tests. Comment- and blank-line-only edits are checked by compiling old and new and comparing bytecode, names and constants by value (set constants compared as sets, so the result does not depend on the hash seed; line numbers, including a class's `__firstlineno__`, are ignored). (A class whose `__eq__` mishandles `None` would break this; in sqlglot at eb11aec only the executor's sort-key wrapper does, and it is never compared with `None`.)
  - Docstring-only edits select nothing: when every changed line sits inside a triple-quoted string, old and new are compiled with module, class and function docstrings removed and compared as above. The scored suite never reads docstrings (`tests/test_docs.py` is not in it; sqlglot only reads `__doc__` to default it to an empty string). Bare string statements, such as attribute docstrings, compile to nothing and are covered by the same comparison.
  - Import crashes: when a change touches code that runs on import (module level, class bodies, decorators and signatures, functions that run during import), the patched sqlglot modules, plus the importers of modules with at most 8 of them, are executed in a forked child on top of the pristine package; the same modules with their pristine text run in a second child. If the patched run raises and the pristine run does not, every test is selected. This catches a class-body `KEYWORDS.pop` of a missing key, a broken SQL template parsed at class definition, and a broken lazy `__getattr__` loader, none of which coverage can see. The image precompiles sqlglot's bytecode so the check does not recompile it at run time.
  - Import-time state check: when the only import-time part of an edit is a function body that coverage saw running during import (not module-level code, class bodies, signatures or new definitions), a forked child imports all of sqlglot with the patch and fingerprints every module and class attribute, function default and closure value; if that matches the pristine fingerprint (computed once per run, in the same process so the hash seed matches), the edit changed nothing that outlives a test and the family fallback is skipped. Dicts and class-only tuples are compared as unordered, since their order follows the hash seed.
  - Readers of changed import-time values: for class-body, module-level and import-time function edits, the state check returns *which* module globals and class attributes (nested classes included) differ after importing the patched package. A changed class attribute selects the tests that read it, from `maps/attrs.{idx,sets}`: built from one pass over the pinned suite in which every plain data attribute of every sqlglot class was replaced by a recording descriptor and every test ran in its own fork (so caches from earlier tests cannot hide a read); reads while a test module is imported count for all its tests. A changed module global selects the tests that run a line naming it. Added or removed values, changed function defaults or closures, descriptors no recorder covered, edited lambdas and failed imports keep the broad selection. The check costs about a CPU second, so it runs only when the broad selection's packed cost exceeds the narrow one by more than 3 s.
  - Key-level reads: class-level dicts (FUNCTIONS, KEYWORDS, TRANSFORMS, ...) were recorded key by key: lookups (`d[k]`, `get`, `in`, including missing keys) per key, and any whole-dict use (iteration, copy, merge, `len`, comparison) as `{ALL}`. The state fingerprint hashes such dicts per key, so a changed, added or removed entry selects who looked that key up plus the whole-dict users. Dicts the recorder did not watch key by key fall back to all their readers.
  - Check runs: when a selection is small (at most 60 tests) and its packed cost exceeds an estimate of the CPU to verify it by 0.5 s, a forked child serves the patched sources from memory and runs exactly those tests with unittest (integration tests enabled, as in the scored suite). If every one passes, nothing the commit selected can fail and the commit runs nothing; any failure, error, skip, import problem or timeout keeps the selection. Only edits to existing .py files under sqlglot/ and tests/ qualify. `unittest` is imported only inside the child: sqlglot records at import whether it is loaded.
  - test_max_nodes stand-in: the 28.8 s test is selected by most parser edits and failed in 2 of 299 labeled commits. When dropping it would save over 4 s and the patch only edits existing .py files (not the test file itself), a child runs the test's statements and assertions at 1/25 scale (width 400, node limits 40,000 and 2,000) on the patched sources, and the same stand-in once on the pinned sources. The test is dropped only if the patched stand-in passes and uses at most 1.5x the pinned stand-in's CPU (+0.3 s), since an edit that slows parsing could time the real test out. Both labeled failures are kept. This is a check, not a proof: an effect that needs more than 400 columns would pass it.
  - Test-file edits select the edited test and the test files that import that module.
  - Fixture edits select the tests that open the fixture.
  - Edits that leave the AST unchanged (comments, blank lines) select nothing.
  - Anything else not understood selects every test.
  - Tests are packed onto 1–8 workers to minimise the busiest worker plus 2 s per worker.

---

## Starter kit notes

A working submission for the Blast Radius track. It runs every test on every commit, so it never misses a failure and scores badly. Your job is to run less without missing.

## The problem

Your image gets a batch of code changes to [sqlglot](https://github.com/tobymao/sqlglot) at commit `eb11aec4be8cb2122afe713a4816662c3f0fb2fa`. For each change, print which tests to run and how to split them across up to 8 workers.

| | |
| --- | --- |
| `/data/commits/<id>.patch` | One unified diff against the pinned commit. Each stands alone: none builds on another. |
| `/data/tests.json` | Every test, numbered, with its file and the seconds it took us to run. |
| stdout | One JSON line a commit: `{"commit": "k7q2", "workers": [[12, 845, 901], [3, 77]]}` |

Each inner list is one worker's test numbers. `"workers": []` runs nothing, which is the right answer for a change that breaks nothing.

## Scoring

Lower is better. [SCORING.md](SCORING.md) has the exact rules; in short:

- A commit costs the recorded seconds of its busiest worker, plus 2 seconds for every worker.
- A commit where you left out a test that fails adds 300 seconds.
- The CPU seconds your program uses are added. Running sqlglot's tests while you decide is allowed, and you pay for the time.
- Your image runs 3 times and the middle total counts.

"Fails" means pytest reports a failure, an error or a timeout for the test, or the test's file cannot be imported. A test with subtests is one test: one failing subtest fails it.

## Try it

You need Docker, make and Python 3.9 or newer. The kit unpacks into the folder you are in, so unpack it into an empty one.

```sh
make sample   # build, run once on the 30 sample commits under the scorer's limits, and score
make check    # the output's format only
```

`make sample` prints your simulated seconds and misses, which are exact for that output. The board adds the CPU seconds your container used and takes the middle of 3 runs, so expect it a little higher. Your program never sees `sample/answers.json`: only the patches and `tests.json` are mounted.

`sample/answers.json` lists the tests each sample commit breaks and the kind of change it is. The practice and final commits are hidden, and so is everything your program prints when it runs on them. A rejected submission tells you which rule it broke and nothing more, so reproduce problems here.

## Submit

Push to a public GitHub repository with the `Dockerfile` at its top, then enter the repository and the commit hash on the track page. The final scores the commit you pick. If you pick none, your best practice submission counts.

## Rules

- Submit your own team's repository. A submission with the same repository, commit or archive as another team's is disqualified, the later one.
- Your build may download only from public package registries: PyPI, npm, Docker Hub, GitHub releases at a fixed tag, and sqlglot’s own repository at the pinned commit. Nothing from an address your team controls. The final rebuilds every image after the cutoff, so anything else could change a submission after the deadline.
- No answers worked out ahead of time and looked up by patch: no table from a patch, or from a kind of edit, to the tests it breaks. Maps built from the pinned commit (coverage, imports, fixtures) are allowed. Top entries are read.
- Only the final counts. The practice board carries no prize.

## How your image runs

- The build has the network, 4 cores, 8 GB of memory and 10 minutes. It may clone sqlglot at the pinned commit, install it and run its tests to make coverage or import maps.
- Your repository may hold 20 MB. Git submodules and Git LFS files are not fetched: clone sqlglot in the `Dockerfile` and commit maps as plain files.
- The run has no network, 2 cores, 4 GB of memory and 3 minutes for the whole batch (4.5 in the final, which has about one and a half times as many commits).
- The disk is read-only. `/tmp` holds 64 MB and is empty at the start of every run.
- The container runs as user `nobody`, with `HOME=/tmp` and at most 256 processes.
- stdout may hold 2 MB. More rejects the run.

## Things that will catch you

- Logs go to stderr. Any line on stdout that is not an answer rejects the run.
- `score.py` names a broken rule by its code, as in `Rejected: not_json on line 1`. The table in SCORING.md says what each code means.
- Files your program needs must be readable by `nobody`. A map written by `root` during the build with mode 600 is not.
- pytest 9 prints a test as `PASSED` beside `SUBFAILED` lines when one of its subtests fails. That test failed.
- sqlglot skips its executor tests unless `SKIP_INTEGRATION=0` is set. We labeled with it set, so those tests can fail.
- `requirements.lock` is every package in the environment we labeled in, on the image the `Dockerfile` starts from. To run sqlglot's tests the way we did, `pip install -r requirements.lock` and put the sqlglot checkout on `PYTHONPATH`; sqlglot's own `pip install -e .[dev]` picks newer versions.
- The scorer is linux/amd64. On an Apple Silicon Mac, build and run with `--platform linux/amd64`, as the Makefile does, or your timings and wheels will differ.
- A change that only adds a comment is not promised to be harmless.
