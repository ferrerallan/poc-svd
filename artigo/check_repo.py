"""
* what? - Checks the repository against the refactoring rules: traceability, the commentator pattern and English code.
* why?  - Keeps the story of the research true: every script is cited in the article and every script follows one style.
* how?  - Parses artigo/index.html, the Python AST and tokens, and the shell scripts; prints every violation and exits 1.

Checks:
  (a) every script and template under doutorado/ and artigo/ is cited in artigo/index.html inside <code data-trace>,
      and every cited path exists (paths inside <code data-legacy> are old names and are not checked)
  (b) every def/class has a 3-line what?/why?/how? docstring and every function body is covered by numbered #region blocks;
      shell scripts start with the same three lines and use numbered "# region N" blocks
  (c) comments, docstrings and identifiers have no accented Latin letters and no word of the Portuguese deny list
Run from the repository root:  python3 artigo/check_repo.py
"""
import ast
import glob
import io
import re
import sys
import tokenize
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ["doutorado", "artigo"]
SKIP_PARTS = {"__pycache__", "versoes", "images", "dados"}
# page and script templates are page content embedded byte for byte in frozen pages: they are cited but not style-checked
TEMPLATES = ("_template.html", "_template.js")
# text that quotes frozen data names is not code language: quoted strings, file names with an extension and snake_case names
QUOTED_OR_DATA = r"'[^']*'|`[^`]*`|\"[^\"]*\"|[\w.-]+\.(json|pt|html|py|sh|csv|log|txt|js|parquet|safetensors|png)\b|\w*_\w*"
DENY = {"para", "uma", "com", "dos", "das", "nao", "pelo", "pela", "cada", "mais", "como", "quando", "porque", "isso", "onde", "ainda",
        "entre", "sobre", "depois", "antes", "treino", "teste", "semente", "sementes", "matriz", "matrizes", "modelo", "modelos",
        "pesos", "peso", "saida", "caixa", "seta", "salvar", "carregar", "resultado", "resultados", "foto", "fotos", "etapas"}


def repo_files():
    """
    * what? - Lists the scripts and templates of the repository that the rules apply to.
    * why?  - The three checks share the same file list.
    * how?  - Walks doutorado/ and artigo/, keeps .py, .sh, .js and template files, skips caches, versions and images.
    """
    #region 1 - Walk the folders
    # what? - Returns a sorted list of Paths.
    # how?  - Any path containing a skipped folder name is ignored.
    found = []
    for d in SCAN_DIRS:
        for p in (ROOT / d).rglob("*"):
            if not p.is_file() or SKIP_PARTS & set(p.relative_to(ROOT).parts):
                continue
            if p.suffix in (".py", ".sh", ".js") or p.name.endswith(TEMPLATES):
                found.append(p)
    return sorted(found)
    #endregion


def check_traceability(files):
    """
    * what? - Checks that every file is cited in the article and every cited path exists.
    * why?  - The article is the single index of the research: scripts that it does not cite are untraceable.
    * how?  - Collects the <code data-trace> paths of artigo/index.html and compares them with the file list.
    """
    #region 1 - Collect the cited paths
    # what? - The set of paths inside <code data-trace>.
    # how?  - Regex over the HTML; the data-legacy elements are ignored on purpose.
    html = (ROOT / "artigo/index.html").read_text(encoding="utf-8")
    cited = {m.strip() for m in re.findall(r"<code[^>]*data-trace[^>]*>([^<]+)</code>", html)}
    problems = []
    #endregion

    #region 2 - Compare with the repository
    # what? - Uncited files and cited paths that do not exist.
    # how?  - A cited path may contain * and then needs at least one match.
    for f in files:
        if str(f.relative_to(ROOT)) not in cited:
            problems.append(f"not cited in artigo/index.html: {f.relative_to(ROOT)}")
    for c in sorted(cited):
        if "*" in c:
            ok = bool(glob.glob(str(ROOT / c)))
        else:
            ok = (ROOT / c).exists()
        if not ok:
            problems.append(f"cited path does not exist: {c}")
    return problems
    #endregion


def regions_of(lines, indent, first, last):
    """
    * what? - Returns the (number, start line, end line) of the #region blocks at a given indentation inside a line range.
    * why?  - The commentator pattern requires every function body to be covered by numbered regions.
    * how?  - Scans for '#region N - title' and the matching '#endregion' at exactly that indentation.
    """
    #region 1 - Scan the lines
    # what? - A list of region intervals, 1-based line numbers.
    # how?  - Regions of nested functions have a deeper indentation and are ignored here; only lines first..last are read.
    pad = " " * indent
    found, open_region = [], None
    for i, line in enumerate(lines, start=1):
        if i < first or i > last:
            continue
        m = re.match(rf"^{pad}#region (\d+) - \S", line)
        if m:
            open_region = (int(m.group(1)), i)
        elif line == f"{pad}#endregion" and open_region:
            found.append((open_region[0], open_region[1], i))
            open_region = None
    return found
    #endregion


def check_python(path):
    """
    * what? - Checks one Python file for the commentator pattern and for English identifiers, comments and docstrings.
    * why?  - Requirements 4 and 5 of the refactoring apply to every script that remains.
    * how?  - AST for docstrings and regions, tokenize for comments and names, a word deny list for Portuguese.
    """
    #region 1 - Parse the file
    # what? - Source, lines and AST.
    # how?  - A syntax error is reported as a problem.
    rel = path.relative_to(ROOT)
    source = path.read_text(encoding="utf-8")
    lines = source.split("\n")
    problems = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"{rel}: syntax error {e}"]
    #endregion

    #region 2 - Check docstrings and regions of every def and class
    # what? - 3-line docstring and full region coverage.
    # how?  - Stub bodies (pass, ..., raise NotImplementedError) need a docstring but no regions.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        where = f"{rel}:{node.lineno} {node.name}"
        if doc is None:
            problems.append(f"{where}: no docstring")
            continue
        parts = doc.split("\n")
        ok = (len(parts) == 5 and parts[0] == "" and parts[4].strip() == ""
              and re.match(r"^\s+\* what\? - \S", parts[1]) and re.match(r"^\s+\* why\?  - \S", parts[2]) and re.match(r"^\s+\* how\?  - \S", parts[3]))
        if not ok:
            problems.append(f"{where}: docstring is not the 3-line what/why/how format")
        if isinstance(node, ast.ClassDef):
            continue
        body = node.body[1:]
        stub = all(isinstance(s, (ast.Pass, ast.Raise)) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)) for s in body)
        if not body or stub:
            continue
        indent = body[0].col_offset
        regs = regions_of(lines, indent, node.lineno, node.end_lineno + 1)
        if [r[0] for r in regs] != list(range(1, len(regs) + 1)) or not regs:
            problems.append(f"{where}: regions are missing or not numbered 1..n")
            continue
        for s in body:
            first = s.decorator_list[0].lineno if getattr(s, "decorator_list", None) else s.lineno
            if not any(a <= first and s.end_lineno <= b for _, a, b in regs):
                problems.append(f"{where}: statement at line {s.lineno} is outside every region")
        for _, a, _b in regs:
            if not (lines[a].lstrip().startswith("# what? - ") and lines[a + 1].lstrip().startswith("# how?  - ")):
                problems.append(f"{where}: region at line {a} lacks the what?/how? lines")
    #endregion

    #region 3 - Check the language of comments, docstrings and identifiers
    # what? - Accented Latin letters and Portuguese deny-list words.
    # how?  - Quoted text, file names and snake_case data names are removed before the word test.
    texts = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            texts.append((tok.start[0], "comment", tok.string))
        elif tok.type == tokenize.NAME:
            texts.append((tok.start[0], "identifier", tok.string))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            d = ast.get_docstring(node, clean=False)
            if d:
                texts.append((getattr(node, "lineno", 1), "docstring", d))
    for line_no, kind, text in texts:
        bad_chars = {c for c in text if ord(c) > 127 and "LATIN" in unicodedata.name(c, "")}
        cleaned = re.sub(QUOTED_OR_DATA, " ", text)
        words = {w.lower() for w in re.findall(r"[A-Za-z]+", cleaned)} & DENY if kind != "identifier" else ({text.lower()} & DENY)
        if bad_chars or words:
            problems.append(f"{rel}:{line_no} {kind} looks Portuguese: {sorted(bad_chars | words)}")
    return problems
    #endregion


def check_shell(path):
    """
    * what? - Checks one shell script for the three-line header, numbered regions and English comments.
    * why?  - The commentator pattern is applied to shell scripts by analogy, with '# region N' markers.
    * how?  - Reads the first lines for the header and scans the region markers; comments go through the language test.
    """
    #region 1 - Check the header and the regions
    # what? - what/why/how lines near the top and sequential region numbers.
    # how?  - Every '# region' needs a matching '# endregion'.
    rel = path.relative_to(ROOT)
    lines = path.read_text(encoding="utf-8").split("\n")
    problems = []
    head = "\n".join(lines[:12])
    for tag in ("* what? - ", "* why?  - ", "* how?  - "):
        if f"# {tag}" not in head:
            problems.append(f"{rel}: header lacks '# {tag.strip()}'")
    numbers = [int(m.group(1)) for line in lines for m in [re.match(r"^# region (\d+) - \S", line)] if m]
    if numbers != list(range(1, len(numbers) + 1)) or not numbers:
        problems.append(f"{rel}: '# region N' blocks are missing or not numbered 1..n")
    if sum(1 for line in lines if line.strip() == "# endregion") != len(numbers):
        problems.append(f"{rel}: '# endregion' count differs from the region count")
    #endregion

    #region 2 - Check the language of the comments
    # what? - Accented letters and deny-list words in comment lines.
    # how?  - Only lines whose first non-blank character is '#'.
    for i, line in enumerate(lines, start=1):
        if line.lstrip().startswith("#"):
            bad_chars = {c for c in line if ord(c) > 127 and "LATIN" in unicodedata.name(c, "")}
            cleaned = re.sub(QUOTED_OR_DATA, " ", line)
            words = {w.lower() for w in re.findall(r"[A-Za-z]+", cleaned)} & DENY
            if bad_chars or words:
                problems.append(f"{rel}:{i} comment looks Portuguese: {sorted(bad_chars | words)}")
    return problems
    #endregion


def main():
    """
    * what? - Runs the three checks over the repository and exits with 0 when clean and 1 otherwise.
    * why?  - One command tells whether the refactoring rules still hold.
    * how?  - Lists the files, runs each check, prints every problem and a one-line summary.
    """
    #region 1 - Run the checks
    # what? - Collects the problems of the three checks.
    # how?  - Templates are only traced, never style-checked.
    files = repo_files()
    problems = check_traceability(files)
    for f in files:
        if f.name.endswith(TEMPLATES):
            continue
        if f.suffix == ".py":
            problems += check_python(f)
        elif f.suffix == ".sh":
            problems += check_shell(f)
    #endregion

    #region 2 - Report
    # what? - Prints the problems and the verdict.
    # how?  - The exit code is what scripts and CI would read.
    for p in problems:
        print(p)
    print(f"{len(files)} files checked, {len(problems)} problem(s)")
    sys.exit(1 if problems else 0)
    #endregion


if __name__ == "__main__":
    main()
