# CLAUDE.md

Read `AGENTS.md` first: it holds the project state, architecture, safety rules, API and changelog.

When you change anything, update `AGENTS.md` in the same commit (feature table, API list if endpoints changed,
one changelog line), run `.venv/bin/pytest -q`, and never commit `.env` or `data/`.
