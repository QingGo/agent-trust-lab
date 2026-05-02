"""ONNX model export for FaithfulnessChecker and AnchoringReasoner.

Provides programmatic functions to export HuggingFace models to ONNX format,
and check whether ONNX models are already cached.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from agent_trust_lab.log import get_logger

logger = get_logger("onnx_setup")

DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "agent-trust-lab", "onnx")


@dataclass
class ExportConfig:
    name: str
    model_name: str
    model_class: Optional["Type[Any]"] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    subdir: str = ""


_EXPORT_CONFIGS: Dict[str, ExportConfig] = {
    "nli": ExportConfig(
        name="nli",
        model_name="roberta-base-mnli",
        subdir="roberta-base-mnli",
    ),
    "embed": ExportConfig(
        name="embed",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"pooling": "mean", "normalize": True},
        subdir="all-MiniLM-L6-v2",
    ),
}


def check_models_available() -> Dict[str, bool]:
    """Check which ONNX models are present in the cache directory.

    Returns:
        Dict with model names as keys and boolean availability as values.
    """
    status: Dict[str, bool] = {}
    for name, config in _EXPORT_CONFIGS.items():
        model_dir = os.path.join(DEFAULT_CACHE, config.subdir)
        model_path = os.path.join(model_dir, "model.onnx")
        status[name] = os.path.exists(model_path)
    return status


def export_model(
    config: ExportConfig,
    output_dir: str,
    hf_token: str = "",
) -> Dict[str, Any]:
    """Export a single HuggingFace model to ONNX format.

    Args:
        config: Export configuration for the model.
        output_dir: Directory to save model.onnx and tokenizer files.
        hf_token: Optional HuggingFace API token.

    Returns:
        Dict with keys name, path, size_mb.
    """
    os.makedirs(output_dir, exist_ok=True)

    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    logger.info("Exporting %s to ONNX -> %s", config.model_name, output_dir)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, token=hf_token or None)

    if config.name == "nli":
        from optimum.onnxruntime import ORTModelForSequenceClassification  # pyright: ignore[reportMissingImports]  # noqa: I001

        model_cls: Any = ORTModelForSequenceClassification
    elif config.name == "embed":
        from optimum.onnxruntime import ORTModelForFeatureExtraction  # pyright: ignore[reportMissingImports]  # noqa: I001

        model_cls = ORTModelForFeatureExtraction
    else:
        raise ValueError(f"Unknown model type: {config.name}")

    model = model_cls.from_pretrained(
        config.model_name,
        export=True,
        token=hf_token or None,
        **config.model_kwargs,
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    model_path = os.path.join(output_dir, "model.onnx")
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    logger.info("Exported %s: %s (%.1f MB)", config.name, model_path, size_mb)

    return {"name": config.name, "path": model_path, "size_mb": size_mb}


def export_all(
    model_filter: str = "all",
    output_dir: Optional[str] = None,
    hf_token: str = "",
) -> List[Dict[str, Any]]:
    """Export ONNX models.

    Args:
        model_filter: Which models to export: "nli", "embed", or "all".
        output_dir: Custom output directory (defaults to DEFAULT_CACHE subdirectories).
        hf_token: Optional HuggingFace API token.

    Returns:
        List of result dicts with name, path, size_mb for each exported model.
    """
    if model_filter not in ("all", "nli", "embed"):
        raise ValueError(f"Unknown model filter: {model_filter}. Use 'nli', 'embed', or 'all'.")

    results: List[Dict[str, Any]] = []
    to_export = list(_EXPORT_CONFIGS.keys()) if model_filter == "all" else [model_filter]

    for name in to_export:
        config = _EXPORT_CONFIGS[name]
        target_dir = output_dir or os.path.join(DEFAULT_CACHE, config.subdir)
        result = export_model(config, target_dir, hf_token)
        results.append(result)

    return results
