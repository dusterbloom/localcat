# Spec: Add repository-wide pre-commit hooks (Python + JS)

Context
- Repo contains a Python backend under `server/` and a Next.js TypeScript frontend under `client/`.
- We want consistent formatting, linting, and basic hygiene checks before commits.
- Prefer Git hook automation via the `pre-commit` framework to keep tooling language-agnostic and easy to run locally and in CI.

Goals
- Add a top-level `.pre-commit-config.yaml` that:
  - Applies general hygiene checks (whitespace, EOF, YAML/TOML/JSON validity, merge conflicts, secrets)
  - Formats Python with Black and lints with Flake8 (per AGENTS_LOCALCAT conventions)
  - Formats JS/TS/JSON/CSS/MD/YAML with Prettier
  - Runs ESLint for the frontend via a local hook (`next lint`)
- Exclude a few very large files from formatting to avoid slow commits.
- Add a short README section with setup instructions.

Non-Goals
- Do not change existing ESLint configuration in `client/`.
- Do not introduce Ruff unless requested; use Black + Flake8 per conventions.
- Do not enforce type-checking (mypy/tsc) at commit time.

Acceptance Criteria
- Running `pre-commit run --all-files` from repo root passes on a clean tree (after any auto-fixes).
- Python files under `server/` and `scripts/` are formatted by Black and checked by Flake8.
- Frontend files under `client/` are formatted by Prettier and linted by ESLint via `next lint`.
- Large files (e.g., `docs/locomo10.json`, `docs/mem0_github_repo.txt`, `server/uv.lock`) are excluded from Prettier.

Deliverables
1) Root `.pre-commit-config.yaml` with:
   - pre-commit-hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, check-json, check-merge-conflict, detect-private-key, check-added-large-files
   - psf/black for Python formatting (targeting `server/` and `scripts/`)
   - pycqa/flake8 for Python lint (targeting `server/` and `scripts/`)
   - mirrors-prettier for general formatting (JS/TS/JSON/CSS/MD/YAML) with excludes
   - local hook to run `next lint` in `client/` (ESLint)
2) Root `.prettierignore` with large-file exclusions and common build artifacts.
3) README update: brief section on installing and using pre-commit.

TDD Guidance
- This change is configuration-focused; no code behavior. TDD not required. Validation is via `pre-commit run --all-files`.

Commands (to be run by Droid Exec)
```bash
# Optional: install pre-commit
pipx install pre-commit || pip install pre-commit

# Install hooks
pre-commit install

# Test on all files
pre-commit run --all-files
```

Owner
- DevOps Automation (via Droid Exec)

Risks
- Initial run may reformat many files; ensure team alignment.
- ESLint hook uses `client`'s devDependencies; requires `npm ci` in `client/`.

Rollback
- Remove `.pre-commit-config.yaml` and `.prettierignore` changes if needed.
