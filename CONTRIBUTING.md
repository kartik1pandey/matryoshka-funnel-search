# Contributing

This is primarily a solo research-reproduction project, but it follows normal
open-source hygiene so the history stays useful and the repo is credible in a
technical interview context.

## Workflow

1. Create a branch per unit of work: `feat/matryoshka-head`, `fix/eval-recall-k`, `docs/architecture`.
2. Keep commits small and in the imperative mood ("Add nested-loss head", not "added" or "adding").
3. Open a PR into `main` even when working solo — it gives CI a chance to run and gives you a reviewable diff later.
4. CI (`.github/workflows/ci.yml`) must pass: ruff, black, mypy, pytest.
5. Squash-merge PRs so `main` history reads as one entry per feature.

## Local setup

```bash
make setup       # editable install + pre-commit hooks
make check       # lint + typecheck + test, same as CI
```

## Commit message convention

Loosely follows [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.

## Adding a new module

Every new module under `src/matryoshka_search/` needs:
- a module-level docstring explaining *why* it exists (not just what it does)
- a corresponding entry in `docs/05_code_walkthrough.md`
- at least one test under `tests/`, unless it requires GPU/dataset download (mark with `@pytest.mark.slow` or `@pytest.mark.gpu`)
