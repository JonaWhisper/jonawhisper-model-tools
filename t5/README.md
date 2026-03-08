# T5 Correction Models

Full pipeline: **Source (PyTorch) → ONNX FP32 → ONNX INT8**

## Models

| ID | Source (PyTorch) | Params | ONNX repo (FP32 + INT8) |
|----|------------------|--------|-------------------------|
| gec-t5-small | [Unbabel/gec-t5_small](https://huggingface.co/Unbabel/gec-t5_small) | 60M | [JonaWhisper/jonawhisper-gec-t5-small-onnx](https://huggingface.co/JonaWhisper/jonawhisper-gec-t5-small-onnx) |
| t5-spell-fr | [fdemelo/t5-base-spell-correction-fr](https://huggingface.co/fdemelo/t5-base-spell-correction-fr) | 220M | [JonaWhisper/jonawhisper-t5-spell-fr-onnx](https://huggingface.co/JonaWhisper/jonawhisper-t5-spell-fr-onnx) |
| flanec-base | [morenolq/flanec-base-cd](https://huggingface.co/morenolq/flanec-base-cd) | 250M | [JonaWhisper/jonawhisper-flanec-base-onnx](https://huggingface.co/JonaWhisper/jonawhisper-flanec-base-onnx) |
| flanec-large | [morenolq/flanec-large-cd](https://huggingface.co/morenolq/flanec-large-cd) | 800M | [JonaWhisper/jonawhisper-flanec-large-onnx](https://huggingface.co/JonaWhisper/jonawhisper-flanec-large-onnx) |

## Usage

```bash
pip install -r t5/requirements.txt

# Full pipeline: source → ONNX FP32 → INT8 (+ upload)
python t5/pipeline.py --upload

# Single model
python t5/pipeline.py --model t5-spell-fr --upload

# Or run steps separately:
python t5/convert.py --model t5-spell-fr    # Step 1: PyTorch → ONNX
python t5/quantize.py --model t5-spell-fr   # Step 2: FP32 → INT8
```

## Adding new models

1. Add an entry to `models.json` with `source` and `repo` fields
2. Trigger the `T5 Pipeline` workflow or run locally
