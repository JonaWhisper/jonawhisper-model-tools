# JonaWhisper Model Tools

Scripts and CI workflows for converting, quantizing, and uploading models used by [JonaWhisper](https://github.com/jplot/jona-whisper).

## T5 Correction Models

Full pipeline: **Source (PyTorch) → ONNX FP32 → ONNX INT8**

### Models

| ID | Source (PyTorch) | Params | ONNX repo (FP32 + INT8) |
|----|------------------|--------|-------------------------|
| gec-t5-small | [Unbabel/gec-t5_small](https://huggingface.co/Unbabel/gec-t5_small) | 60M | [JonaWhisper/jonawhisper-gec-t5-small-onnx](https://huggingface.co/JonaWhisper/jonawhisper-gec-t5-small-onnx) |
| t5-spell-fr | [fdemelo/t5-base-spell-correction-fr](https://huggingface.co/fdemelo/t5-base-spell-correction-fr) | 220M | [JonaWhisper/jonawhisper-t5-spell-fr-onnx](https://huggingface.co/JonaWhisper/jonawhisper-t5-spell-fr-onnx) |
| flanec-base | [morenolq/flanec-base-cd](https://huggingface.co/morenolq/flanec-base-cd) | 250M | [JonaWhisper/jonawhisper-flanec-base-onnx](https://huggingface.co/JonaWhisper/jonawhisper-flanec-base-onnx) |
| flanec-large | [morenolq/flanec-large-cd](https://huggingface.co/morenolq/flanec-large-cd) | 800M | [JonaWhisper/jonawhisper-flanec-large-onnx](https://huggingface.co/JonaWhisper/jonawhisper-flanec-large-onnx) |

### Local usage

Individual steps or full pipeline:

```bash
pip install optimum[exporters] onnxruntime huggingface-hub

# Full pipeline: source → ONNX FP32 → INT8 (+ upload)
python t5/pipeline.py --upload

# Single model
python t5/pipeline.py --model t5-spell-fr --upload

# Or run steps separately:
python t5/convert.py --model t5-spell-fr    # Step 1: PyTorch → ONNX
python t5/quantize.py --model t5-spell-fr   # Step 2: FP32 → INT8
```

### CI

The `T5 Pipeline` workflow runs the full pipeline (convert + quantize + upload) and can be triggered manually from the Actions tab. Requires a `HF_TOKEN` secret.

## Adding new models

1. Find the source model on HuggingFace (PyTorch/safetensors)
2. Create the target repo on HuggingFace under the `JonaWhisper` org
3. Add an entry to `t5/models.json` with `source` and `repo` fields
4. Trigger the CI workflow or run `python t5/pipeline.py --model <id> --upload`

## License

GPL-3.0-or-later
