# Contributing

## Commits

This project uses [Conventional Commits](https://www.conventionalcommits.org/).

Format: `type(scope): description`

### Types

- `feat` — New feature (script, model, workflow)
- `fix` — Bug fix
- `refactor` — Code change that neither fixes a bug nor adds a feature
- `chore` — Maintenance (dependencies, CI config)
- `docs` — Documentation only

### Scopes

- `ci` — GitHub Actions workflows
- `t5` — T5 correction models
- `pcs` — PCS punctuation models
- `bert` — BERT punctuation models

### Examples

```
feat(t5): add new correction model xyz
feat(pcs): add tokenizer conversion script
fix(ci): update Python version for compatibility
refactor(t5): use Python API instead of CLI subprocess
docs: update README with new model
chore: update dependencies
```

## Pull Requests

- One feature/fix per PR
- PR title follows the same conventional commit format
- Describe what changed and why in the PR body
- Link to relevant HuggingFace model repos if applicable

## Adding a New Model

1. Find the source model on HuggingFace (PyTorch/safetensors)
2. Create the target ONNX repo on HuggingFace under `JonaWhisper/`
3. Add an entry to `t5/models.json` with `source` and `repo` fields
4. Test locally: `python t5/pipeline.py --model <id>`
5. Open a PR

## Development

```bash
pip install -r requirements.txt
python t5/pipeline.py --help
```
