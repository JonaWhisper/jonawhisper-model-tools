# JonaWhisper Model Tools

Scripts and CI workflows for building, converting, and uploading models used by [JonaWhisper](https://github.com/jplot/jona-whisper). Each model type lives in its own directory with its own `requirements.txt`.

## Pipelines

| Directory | Models | CI Workflow | HuggingFace |
|-----------|--------|-------------|-------------|
| [`t5/`](t5/) | T5 correction (GEC, spell) | `T5 Pipeline` | Per-model repos under `JonaWhisper/` |
| [`kenlm/`](kenlm/) | KenLM n-gram language models | `Build KenLM models` | [JonaWhisper/kenlm-models](https://huggingface.co/JonaWhisper/kenlm-models) |

## CI

All workflows are triggered manually from the Actions tab and require a `HF_TOKEN` secret for HuggingFace uploads.

## License

GPL-3.0-or-later
