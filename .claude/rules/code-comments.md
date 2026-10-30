# Code comments and docstrings

Python code follows PEP 8 for comments and layout, and PEP 257 for docstrings, written in the Google style (`Args:`, `Returns:`, `Raises:`). When black is set up in the repo it owns all formatting: never hand-format against it, and take the line length from `pyproject.toml`. The rules below decide where comments are allowed and what a docstring holds.

## Comments

Allowed in code: docstrings, numbered flow steps inside multi-step functions, and tool directives (`# noqa: <code>`, `# pylint: disable=<name>`, `# type: ignore[<code>]`, `# fmt: off` and `# fmt: on`, the shebang line). Nothing else. This covers every file type: source, tests, .env files, Dockerfiles, compose files, CI YAML, requirements files.

Never write:

- Narration of what the next line does
- Notes explaining a fix or justifying a choice to the reviewer ("ported from X", "covered elsewhere by Y", arithmetic walk-throughs)
- Descriptions of what the code does not do
- Bug numbers, incident stories, or "verified live" notes
- Policy or rationale blocks. Rationale goes to a cortex ADR, and writing it there means deleting it from the code.

A directive names its code or check: `# noqa: E501`, never a bare `# noqa`. Config and infra files get no per-line notes. Annotate only a line that would look like a mistake without one.

Before finishing any file, scan every comment and delete any that is not a docstring, a numbered step or a tool directive.

## Docstrings

- In triple double quotes, on public functions, classes and methods. Not on constants, dataclass or model fields, private helpers whose name says what they do, or test functions. A module docstring is optional.
- A summary line of one sentence, ending with a full stop. A second paragraph only for a caller contract.
- `Raises:` lists every exception a public function raises on purpose.
- `Args:` only when the name and type hint do not explain the argument, such as callbacks and flags.
- `Returns:` only when the return type hint does not say it.
- Types live in the type hints, never in the docstring.
- No examples or doctests. No history, no rationale.

## Numbered steps

One line per step, stating the rule the step enforces:

```python
# 1. Reject an image larger than the configured limit.
# 2. Decode in memory; nothing is written to disk.
```

- A second line only to say what a status means.
- Fire-and-forget calls get one line saying the failure is non-fatal.
- Comment a guard clause only when it encodes a business rule that the names do not show.
- Skip numbering on functions with one or two steps.
