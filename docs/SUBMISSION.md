# What the submission ships

The submission build (`submission-v6-rc`: `gen-integration` `52ed366`, generator off, plus the hardening fixes). The Docker image
copies only the paths the Dockerfile names; `.dockerignore` keeps everything else out of the build
context.

## Ships

| Path | Why |
|---|---|
| `Dockerfile`, `.dockerignore`, `requirements.txt` | the build |
| `engine/` (all `.py` except `gen_candidates.py`, `critic_score.py`, `search.py`) | the engine; `engine/main.py` is the provided orchestrator |
| `engine/data/layout_library.json` (771,697 bytes) | Crello-derived geometry and roles only (CDLA-Permissive-2.0, see README) |
| `eval/*.py` (top level only, except the generator/critic tooling listed in `.dockerignore`) | `check.py` and `score.py` are imported by the engine at run time; the rest is our evaluation tooling |
| `eval/canvas_catalogue.json` | target formats used by the tooling (repository only; no Crello content) |
| `renderer/` | the provided renderer |
| `brand_kit/` | the provided fonts and sample assets |
| `chameleon_design_schema_v0.9.json` | the provided schema |
| `input/chameleon_sample_*.json` (3 files) | the provided sample tasks |
| `output/.gitkeep` | the mounted output folder |
| `README.md`, `RULES.md`, `FAQ.md` | documentation and the provided rules |
| `requirements-dev.txt`, `pytest.ini`, `eval/tests/` (the tests that need no local data) | tests (repository only; not in the image) |

## Must not ship

| Path | Why |
|---|---|
| `eval/results/labels.json`, any `eval/results/**/labels.json`, `.labels.*` | human labels (test set) |
| `eval/results/*_tags.json` | tags from the diagnosis and A/B pages |
| `eval/results/crello-diagnose/` | diagnosis and A/B pages: Crello renders |
| `eval/results/crello-*/outputs/`, `renders/`, `contact_sheet.png`, `eval/results/outputs/`, `renders/` | engine outputs and renders of Crello designs |
| `eval/results/**/cases.csv`, `summary.md` | benchmark tables keyed by Crello design ids (scores only; tracked in git today, leave out of a public submission) |
| `eval/crello_testset.json`, `eval/crello_quick.json`, `eval/crello_structure.json`, `eval/crello_fonts.json` | Crello design id lists and font maps for our test sets |
| `NOTES.md` | working notes; names and ids of Crello test designs |
| `data/` (outside the repo: `crello_chameleon/`, `generator/`, labelling bundles, `lockbox_ids.txt`) | the converted dataset, labels, the lockbox, big data |
| `engine/data/generator.onnx`, `engine/data/critic.onnx`, any `*.onnx` | model weights; not part of the submission |
| `critic/`, `generator/`, `engine/gen_candidates.py`, `engine/critic_score.py`, `engine/search.py` | generator and critic code (opt-in, off); not part of the submission |
| `cr-renderer/`, `.venv/`, `__pycache__/`, `.pytest_cache/` | local checkouts and caches |
| `output/*` (except `.gitkeep`) | generated outputs |
| `CLAUDE.md`, `docs/agents/` | agent workflow notes |

The image already contains none of the "must not ship" paths except the tooling scripts in `eval/*.py`
(code only, no data). If the repository itself is handed in, remove or exclude the rows above first.
