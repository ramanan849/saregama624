"""Repository conventions that keep the code readable and portable.

* every module, class and function (private and nested ones included) has a
  docstring, so new code cannot land undocumented;
* no source file hard-codes a machine-specific absolute path: data locations
  come from :mod:`saregama.paths` (environment variables or the project layout).
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("saregama", "annotator_model", "tests")


def _sources():
    """Every Python file of the checked packages."""
    for pkg in PACKAGES:
        yield from sorted((ROOT / pkg).rglob("*.py"))


def test_everything_has_a_docstring():
    """List every undocumented definition, not just the first one."""
    missing = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(ROOT)
        if not ast.get_docstring(tree):
            missing.append(f"{rel}: module")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and not ast.get_docstring(node):
                missing.append(f"{rel}:{node.lineno} {node.name}")
    assert not missing, "undocumented:\n" + "\n".join(missing)


def test_no_machine_specific_paths():
    """Absolute home / user paths would break the code on any other machine."""
    pattern = re.compile(r"""["'](/home/|/Users/|[A-Z]:\\\\)""")
    hits = [f"{p.relative_to(ROOT)}:{i}" for p in _sources()
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if pattern.search(line)]
    assert not hits, "hard-coded paths:\n" + "\n".join(hits)
