"""Structural checks on main.tex for machines without a LaTeX install.

    python paper/lint_tex.py

Not a substitute for compiling: it catches undefined references and
citations, missing figure files, unbalanced braces and environments.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = (HERE / "main.tex").read_text(encoding="utf-8")
body = "\n".join(re.split(r"(?<!\\)%", line)[0] for line in s.splitlines())

opens = len(re.findall(r"(?<!\\)\{", body))
closes = len(re.findall(r"(?<!\\)\}", body))
print("braces balanced:", opens == closes, f"({opens} open, {closes} close)")

labels = set(re.findall(r"\\label\{([^}]+)\}", body))
refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
print("undefined refs:", sorted(refs - labels) or "none")
print("unreferenced labels:", sorted(labels - refs) or "none")

cites = {k.strip() for grp in re.findall(r"\\cite\{([^}]+)\}", body) for k in grp.split(",")}
bib = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
print("undefined cites:", sorted(cites - bib) or "none")
print("uncited bibitems:", sorted(bib - cites) or "none")

figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
print("figures present:", {f: (HERE / f).exists() for f in figs})

stack, ok = [], True
for kind, name in re.findall(r"\\(begin|end)\{(\w+)\}", body):
    if kind == "begin":
        stack.append(name)
    elif not stack or stack.pop() != name:
        ok = False
print("environments balanced:", ok and not stack)

print("TODO markers:", len(re.findall(r"\[TODO", s)))
words = re.sub(r"\\[a-zA-Z]+\*?|[{}$\\]", " ", body)
print("approx. words:", len(words.split()))
