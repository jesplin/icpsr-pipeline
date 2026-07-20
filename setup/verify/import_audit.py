#!/usr/bin/env python3
"""Static check: does every lab install the third-party packages it imports?

For each lab this collects the packages it imports (via AST), subtracts the
standard library, the local ``course_utils`` helper, and everything the lab
itself ``!pip install``s, and flags the remainder -- packages the lab USES but
never INSTALLS. This is the "Lab 1 forgot to install openai" class of bug, and
unlike actually running the notebooks it catches them deterministically, no
matter how rich the current environment happens to be.

It intentionally OVER-reports transitively-satisfied deps (e.g. numpy, which
ships with pandas; shapely, which ships with geopandas). Treat the output as a
list to triage, not a list of confirmed bugs -- run ``run_labs.py`` to see what
actually breaks.

Usage
-----
    python setup/verify/import_audit.py
    python setup/verify/import_audit.py --dir solutions
"""
import ast
import re
import sys
from pathlib import Path

import jupytext

REPO = Path(__file__).resolve().parents[2]

# import name -> pip/distribution name, only where they differ
IMPORT_TO_PIP = {
    "bs4": "beautifulsoup4", "PIL": "pillow", "sklearn": "scikit-learn",
    "cv2": "opencv-python", "yaml": "pyyaml", "skimage": "scikit-image",
    "dateutil": "python-dateutil",
}
LOCAL_MODULES = {"course_utils"}     # repo-local helper, not pip-installable
ALLOWLIST = {"google"}               # google.colab -- only present on Colab
STDLIB = set(sys.stdlib_module_names)


def sanitize(src):
    """Neutralize IPython shell/magic lines (!..., %...) so ast.parse works,
    preserving indentation so `if:`/`for:` blocks stay valid."""
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped[:1] in "!%":
            out.append(line[:len(line) - len(stripped)] + "pass")
        else:
            out.append(line)
    return "\n".join(out)


def imports_and_installs(path):
    """Return (imported top-level packages, pip-installed names) for one .py."""
    nb = jupytext.read(path)
    imports, installs = set(), set()
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        # pip installs: scan the raw source
        for line in cell.source.splitlines():
            m = re.search(r"[!%]\s*pip\s+install\s+(.*)", line)
            if not m:
                continue
            for tok in m.group(1).split():
                if tok.startswith("-"):
                    continue
                name = re.split(r"[<>=\[]", tok)[0].strip()
                if name:
                    installs.add(name.lower())
        # imports: parse each cell independently (tolerant of odd cells)
        try:
            tree = ast.parse(sanitize(cell.source))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.add(node.module.split(".")[0])
    return imports, installs


def main():
    subdir = "labs"
    args = sys.argv[1:]
    if "--dir" in args:
        subdir = args[args.index("--dir") + 1]
    work_dir = REPO / subdir

    print(f"{'file':32} {'USES but does not install':40}")
    print("-" * 74)
    any_flag = False
    for path in sorted(work_dir.glob("*.py")):
        imports, installs = imports_and_installs(path)
        missing = []
        for imp in sorted(imports):
            if imp in STDLIB or imp in LOCAL_MODULES or imp in ALLOWLIST:
                continue
            pip_name = IMPORT_TO_PIP.get(imp, imp).lower()
            if pip_name in installs or imp.lower() in installs:
                continue
            missing.append(imp)
        # course_utils.chat()/chat_image() import openai lazily at call time
        src = path.read_text()
        needs_openai = ("course_utils" in src
                        and re.search(r"\bchat(_image)?\(", src)
                        and "openai" not in installs)
        note = ""
        if needs_openai and "openai" not in [m.lower() for m in missing]:
            note = "  [+openai, used via course_utils.chat, not installed]"
            any_flag = True
        if missing:
            any_flag = True
        print(f"{path.name:32} {', '.join(missing) or '-':40}{note}")
    print("-" * 74)
    if not any_flag:
        print("No missing-install issues found.")
    print("Note: transitive deps (numpy<-pandas, shapely<-geopandas, "
          "tqdm<-openai) may show up here but are satisfied at install time.")


if __name__ == "__main__":
    main()
