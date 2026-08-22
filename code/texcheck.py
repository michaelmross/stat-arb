"""Structural integrity check on a LaTeX source file.

NOT PART OF THE PIPELINE. The note's TeX source is not kept in this
repository, so nothing here depends on this script and no workflow step
runs it. It is retained only because it is self-contained and costs
nothing: point it at a .tex file elsewhere and it will report missing
figure targets, dangling \\ref targets, uncited bibitems, and unbalanced
environments or braces. It is not a compiler.

    python code/texcheck.py path/to/note.tex
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path


def check(tex_path: Path) -> bool:
    s = tex_path.read_text(encoding="utf-8")
    ok = True

    def report(name, bad, detail=""):
        nonlocal ok
        if bad:
            ok = False
        print(f"  [{'XX' if bad else 'OK'}] {name}"
              f"{(': ' + detail) if detail else ''}")

    figs = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', s)
    missing = [f for f in figs if not (tex_path.parent / f).exists()]
    report(f"{len(figs)} figure targets exist", missing, ", ".join(missing))

    labs = set(re.findall(r'\\label\{([^}]+)\}', s))
    refs = set(re.findall(r'\\(?:eq|page|auto)?ref\{([^}]+)\}', s))
    report(f"{len(refs)} refs resolve against {len(labs)} labels",
           sorted(refs - labs), ", ".join(sorted(refs - labs)))
    print(f"  [--] unused labels: {', '.join(sorted(labs - refs)) or 'none'}")

    cites = {c.strip() for g in re.findall(r'\\cite[tp]?\{([^}]+)\}', s)
             for c in g.split(",")}
    bibs = set(re.findall(r'\\bibitem\{([^}]+)\}', s))
    if cites or bibs:
        report(f"{len(cites)} citations have bibitems", sorted(cites - bibs),
               ", ".join(sorted(cites - bibs)))

    b = Counter(re.findall(r'\\begin\{([^}]+)\}', s))
    e = Counter(re.findall(r'\\end\{([^}]+)\}', s))
    bad_env = {k: (b[k], e[k]) for k in set(b) | set(e) if b[k] != e[k]}
    report("environments balanced", bad_env, str(bad_env))

    depth = 0
    for i, ch in enumerate(s):
        esc = i > 0 and s[i - 1] == "\\"
        if ch == "{" and not esc:
            depth += 1
        elif ch == "}" and not esc:
            depth -= 1
    report("braces balanced", depth != 0, f"final depth {depth}")
    return ok


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        sys.exit("usage: python code/texcheck.py path/to/note.tex\n"
                 "(the note's source is not part of this repository)")
    tex = Path(argv[0])
    if not tex.exists():
        sys.exit(f"no such file: {tex}")
    print(f"checking {tex}")
    ok = check(tex)
    print(f"\n{'PASS' if ok else 'ISSUES FOUND'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
