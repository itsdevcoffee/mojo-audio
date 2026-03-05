# HuBERT x MAX Engine Experiment

Validates whether MAX Engine's InferenceSession can run HuBERT inference
from an ONNX-exported model and benchmarks it against PyTorch.

## Scripts (run in order)

1. `01_export_onnx.py` — Download HuBERT, export to ONNX
2. `02_validate_onnx.py` — Validate ONNX output matches PyTorch (onnxruntime)
3. `03_max_inference.py` — Load ONNX via MAX Engine, verify output
4. `04_benchmark.py` — Benchmark all backends, print comparison table
5. `05_results.md` — Findings (written after running benchmark)

## Run all

```bash
pixi run python experiments/hubert-max/01_export_onnx.py
pixi run python experiments/hubert-max/02_validate_onnx.py
pixi run python experiments/hubert-max/03_max_inference.py
pixi run python experiments/hubert-max/04_benchmark.py
```
