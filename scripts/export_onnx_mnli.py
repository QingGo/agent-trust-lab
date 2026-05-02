#!/usr/bin/env python
"""Export roberta-base-mnli to ONNX format and save tokenizer for FaithfulnessChecker.

Usage:
    python scripts/export_onnx_mnli.py [--output-dir ~/.cache/agent-trust-lab/onnx/roberta-base-mnli]
    python scripts/export_onnx_mnli.py --hf-mirror https://hf-mirror.com

Requirements:
    pip install optimum[onnxruntime] transformers torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    (torch may not be available for Python 3.13 on x86_64 macOS — run on Linux or Apple Silicon)
"""

import argparse
import os
import sys


def export_model(model_name: str, output_dir: str, hf_token: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)

    from transformers import AutoConfig, AutoTokenizer

    print(f"Exporting {model_name} to ONNX...")
    print(f"Output directory: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token or None)
    config = AutoConfig.from_pretrained(model_name, token=hf_token or None)

    from optimum.onnxruntime import ORTModelForSequenceClassification

    model = ORTModelForSequenceClassification.from_pretrained(
        model_name,
        export=True,
        token=hf_token or None,
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    model_path = os.path.join(output_dir, "model.onnx")
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")

    print(f"Exported model: {model_path} ({os.path.getsize(model_path) / 1024 / 1024:.1f} MB)")
    print(f"Exported tokenizer: {tokenizer_path}")
    print(f"Model type: {config.model_type}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export roberta-base-mnli to ONNX")
    parser.add_argument(
        "--model-name",
        default=os.environ.get("ONNX_MODEL_NAME", "roberta-base-mnli"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "ONNX_OUTPUT_DIR",
            os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "agent-trust-lab",
                "onnx",
                "roberta-base-mnli",
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
