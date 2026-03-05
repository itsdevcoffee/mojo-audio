#!/usr/bin/env python3
"""
Export HuBERT base to ONNX with dynamic input shapes.

Downloads facebook/hubert-base-ls960 from HuggingFace on first run (~360MB).
Outputs: experiments/hubert-max/hubert_base.onnx

This is the model used in RVC v2 for content encoding.
The HuggingFace version is architecturally identical to lj1995/hubert_base.pt.
"""

import torch
import numpy as np
from pathlib import Path
from transformers import HubertModel

OUTPUT_PATH = Path(__file__).parent / "hubert_base.onnx"
MODEL_ID = "facebook/hubert-base-ls960"

# Synthetic 1-second audio at 16kHz (batch=1)
DUMMY_INPUT = torch.zeros(1, 16000)


def export():
    print(f"Loading {MODEL_ID} ...")
    model = HubertModel.from_pretrained(MODEL_ID)
    model.eval()

    print("Exporting to ONNX ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (DUMMY_INPUT,),
            str(OUTPUT_PATH),
            input_names=["input_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "time_frames"},
            },
            opset_version=17,
            do_constant_folding=True,
            # PyTorch 2.10 defaults to dynamo=True which requires onnxscript.
            # Use legacy TorchScript exporter (dynamo=False) which works without it.
            dynamo=False,
        )

    size_mb = OUTPUT_PATH.stat().st_size / 1_000_000
    print(f"Exported: {OUTPUT_PATH} ({size_mb:.1f} MB)")

    # Quick shape check
    import onnx
    model_proto = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(model_proto)
    print("ONNX model validated (checker passed)")

    # Print input/output info
    print("\nModel inputs:")
    for inp in model_proto.graph.input:
        print(f"  {inp.name}: {[d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    print("Model outputs:")
    for out in model_proto.graph.output:
        print(f"  {out.name}: {[d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]}")


if __name__ == "__main__":
    if OUTPUT_PATH.exists():
        print(f"ONNX already exists at {OUTPUT_PATH}, skipping export.")
        print("Delete it to re-export.")
    else:
        export()
