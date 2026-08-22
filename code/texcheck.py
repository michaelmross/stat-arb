"""Structural integrity check on the note's LaTeX source.

Not a compiler -- no TeX toolchain is assumed present. This catches the
failure modes that would break a build or silently produce '??' in the
PDF: missing figure files, dangling \\ref targets, uncited bibitems, and
unbalanced environments or braces.
"""

import re
import sys
from collections import Counter
from pathlib import Path

TEX = Path(sys.argv[1] if len(sys.argv) > 1 else "etf_pairs_note.tex")
s = TEX.read_text(encoding="utf-8")
ok = True


def report(name, bad, detail=""):
    global ok
    if bad:
        ok = False
    print(f"  [{'XX' if bad else 'OK'}] {name}{(': ' + detail) if detail else ''}")


# --- figures ---------------------------------------------------------
figs = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', s)
missing = [f for f in figs if not (TEX.parent / f).exists()]
report(f"{len(figs)} figure targets exist", missing, ", ".join(missing))
onDisk = sorted(p.name for p in TEX.parent.glob("fig_*.pdf"))
unref = [f for f in onDisk if f not in figs]
print(f"  [--] generated but unreferenced: {', '.join(unref) or 'none'}")

# --- cross references ------------------------------------------------
labs = set(re.findall(r'\\label\{([^}]+)\}', s))
refs = set(re.findall(r'\\(?:eq|page|auto)?ref\{([^}]+)\}', s))
report(f"{len(refs)} refs resolve against {len(labs)} labels",
       sorted(refs - labs), ", ".join(sorted(refs - labs)))
print(f"  [--] unused labels: {', '.join(sorted(labs - refs)) or 'none'}")

# --- citations -------------------------------------------------------
cites = {c.strip() for g in re.findall(r'\\cite[tp]?\{([^}]+)\}', s)
         for c in g.split(",")}
bibs = set(re.findall(r'\\bibitem\{([^}]+)\}', s))
if cites or bibs:
    report(f"{len(cites)} citations have bibitems", sorted(cites - bibs),
           ", ".join(sorted(cites - bibs)))

# --- environments and braces -----------------------------------------
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

print(f"\n{'PASS' if ok else 'ISSUES FOUND'}")
sys.exit(0 if ok else 1)
