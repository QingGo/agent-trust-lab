#!/usr/bin/env python
"""Export all-MiniLM-L6-v2 sentence embedding model to ONNX for AnchoringReasoner.

Usage:
    python scripts/export_onnx_embedding.py [--output-dir ~/.cache/agent-trust-lab/onnx/all-MiniLM-L6-v2]
    python scripts/export_onnx_embedding.py --hf-mirror https://hf-mirror.com

Or use the CLI:
    agent-trust-lab setup-onnx --model embed

Requirements:
    pip install optimum[onnxruntime] transformers torch sentence-transformers --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    (torch may not be available for Python 3.13 on x86_64 macOS — run on Linux or Apple Silicon)
"""

import argparse
import os
import sys


def export_model(model_name: str, output_dir: str, hf_token: str = "") -> None:
    from agent_trust_lab.onnx_setup import _EXPORT_CONFIGS, export_model as _export

    config = _EXPORT_CONFIGS["embed"]
    result = _export(config, output_dir, hf_token)
    print(f"Exported model: {result['path']} ({result['size_mb']:.1f} MB)")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all-MiniLM-L6-v2 to ONNX")
    parser.add_argument(
        "--model-name",
        default=os.environ.get("ONNX_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "ONNX_EMBED_DIR",
            os.path.join(
                os.path.expanduser("~"),
                ".cache", "agent-trust-lab", "onnx", "all-MiniLM-L6-v2",
            ),
        ),
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--hf-mirror", default=os.environ.get("HF_ENDPOINT", ""))

    args = parser.parse_args()

    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = args.hf_mirror

    export_model(args.model_name, args.output_dir, args.hf_token)


if __name__ == "__main__":
    main()
