# Contributing to Futurnal

Thanks for contributing.

## Development Setup

```bash
git clone https://github.com/ChrisRPL/Futurnal.git
cd Futurnal

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

Desktop shell:

```bash
cd desktop
npm install
```

## Before Opening a PR

Run the quality gate locally when possible:

```bash
# from repo root
pytest tests/ -v --ignore=tests/performance/
ruff check src tests
mypy src/futurnal --ignore-missing-imports
```

Desktop checks:

```bash
cd desktop
npm run build
npm test
```

## Pull Request Guidance

- Keep changes scoped and reviewable.
- Add or update tests when behavior changes.
- Update docs for user-visible or API changes.
- Use clear commit messages (Conventional Commits preferred).

## Reporting Bugs

Open a GitHub issue with:

- Expected behavior
- Actual behavior
- Repro steps
- Environment info (OS, Python version, model/runtime)
