# Future Work

## Model Catalog Generation

- Keep using `models.json` as the current source of truth for now.
- In a future iteration, add a small generation pipeline so model metadata is maintained from a higher-level source file and emitted into `models.json` or a generated Python artifact.
- The goal is not to replace the current `models.json` workflow immediately, but to make large-scale model metadata updates safer and more consistent when the catalog grows.
- A minimal future version could look like:
  - source file: `models.catalog.json` or `models.catalog.yaml`
  - generator script: `tools/generate_models.py`
  - generated output: current `models.json`
- When we do this, we should also add consistency checks in tests or CI so source and generated output cannot silently drift.
