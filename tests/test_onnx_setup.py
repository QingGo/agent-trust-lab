import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_trust_lab.onnx_setup import (
    _EXPORT_CONFIGS,
    ExportConfig,
    check_models_available,
    export_all,
    export_model,
)


@pytest.fixture(autouse=True)
def _mock_optional_deps(monkeypatch):
    """Preload mock modules so lazy imports inside export_model() succeed."""
    mock_transformers = MagicMock()
    mock_optimum = MagicMock()
    mock_optimum.onnxruntime = MagicMock()

    nli_model_cls = MagicMock()
    nli_model = MagicMock()
    nli_model.save_pretrained = MagicMock()
    nli_model_cls.from_pretrained.return_value = nli_model

    embed_model_cls = MagicMock()
    embed_model = MagicMock()
    embed_model.save_pretrained = MagicMock()
    embed_model_cls.from_pretrained.return_value = embed_model

    mock_tokenizer = MagicMock()
    mock_tokenizer.save_pretrained = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    mock_optimum.onnxruntime.ORTModelForFeatureExtraction = embed_model_cls
    mock_optimum.onnxruntime.ORTModelForSequenceClassification = nli_model_cls
    mock_transformers.AutoTokenizer = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained = MagicMock(
        return_value=mock_tokenizer
    )
    sys.modules["transformers"] = mock_transformers
    sys.modules["optimum"] = mock_optimum
    sys.modules["optimum.onnxruntime"] = mock_optimum.onnxruntime

    real_getsize = os.path.getsize
    monkeypatch.setattr(os.path, "getsize", lambda p: 52428800)

    yield

    monkeypatch.setattr(os.path, "getsize", real_getsize)
    sys.modules.pop("transformers", None)
    sys.modules.pop("optimum", None)
    sys.modules.pop("optimum.onnxruntime", None)


class TestExportConfig:
    def test_nli_config(self):
        config = _EXPORT_CONFIGS["nli"]
        assert config.name == "nli"
        assert config.model_name == "roberta-base-mnli"
        assert config.subdir == "roberta-base-mnli"
        assert config.model_kwargs == {}

    def test_embed_config(self):
        config = _EXPORT_CONFIGS["embed"]
        assert config.name == "embed"
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.subdir == "all-MiniLM-L6-v2"
        assert config.model_kwargs == {"pooling": "mean", "normalize": True}

    def test_config_has_both_models(self):
        assert set(_EXPORT_CONFIGS.keys()) == {"nli", "embed"}


class TestCheckModelsAvailable:
    def test_all_missing_when_cache_empty(self, tmp_path, monkeypatch):
        cache = str(tmp_path / "onnx")
        monkeypatch.setattr(
            "agent_trust_lab.onnx_setup.DEFAULT_CACHE", cache
        )
        status = check_models_available()
        assert status == {"nli": False, "embed": False}

    def test_all_present_when_files_exist(self, tmp_path, monkeypatch):
        cache = str(tmp_path / "onnx")
        monkeypatch.setattr(
            "agent_trust_lab.onnx_setup.DEFAULT_CACHE", cache
        )
        for config in _EXPORT_CONFIGS.values():
            model_dir = os.path.join(cache, config.subdir)
            os.makedirs(model_dir, exist_ok=True)
            Path(os.path.join(model_dir, "model.onnx")).touch()
        status = check_models_available()
        assert status == {"nli": True, "embed": True}

    def test_partial_availability(self, tmp_path, monkeypatch):
        cache = str(tmp_path / "onnx")
        monkeypatch.setattr(
            "agent_trust_lab.onnx_setup.DEFAULT_CACHE", cache
        )
        nli_dir = os.path.join(cache, "roberta-base-mnli")
        os.makedirs(nli_dir, exist_ok=True)
        Path(os.path.join(nli_dir, "model.onnx")).touch()
        status = check_models_available()
        assert status == {"nli": True, "embed": False}


class TestExportModel:
    def test_export_nli_model(self):
        model = export_model(_EXPORT_CONFIGS["nli"], "/tmp/test_onnx_nli")

        assert model["name"] == "nli"
        assert model["path"] == "/tmp/test_onnx_nli/model.onnx"
        assert "size_mb" in model

    def test_export_embed_model(self):
        model = export_model(_EXPORT_CONFIGS["embed"], "/tmp/test_onnx_embed")

        assert model["name"] == "embed"

    def test_export_model_creates_output_dir(self, tmp_path):
        output_dir = str(tmp_path / "subdir" / "deep")
        export_model(_EXPORT_CONFIGS["nli"], output_dir)
        assert os.path.isdir(output_dir)

    def test_export_model_raises_for_unknown_model_type(self):
        config = ExportConfig(name="unknown", model_name="bad/model", subdir="bad")
        with pytest.raises(ValueError, match="Unknown model type"):
            export_model(config, "/tmp/test_onnx")


class TestExportAll:
    def test_export_all_both_models(self):
        mock_results = [
            {"name": "nli", "path": "/tmp/nli/model.onnx", "size_mb": 500.0},
            {"name": "embed", "path": "/tmp/embed/model.onnx", "size_mb": 90.0},
        ]

        with patch(
            "agent_trust_lab.onnx_setup.export_model", side_effect=mock_results
        ) as mock_export:
            results = export_all("all")

        assert len(results) == 2
        assert results[0]["name"] == "nli"
        assert results[1]["name"] == "embed"
        assert mock_export.call_count == 2

    def test_export_all_nli_only(self):
        mock_result = {"name": "nli", "path": "/tmp/nli/model.onnx", "size_mb": 500.0}

        with patch(
            "agent_trust_lab.onnx_setup.export_model", return_value=mock_result
        ) as mock_export:
            results = export_all("nli")

        assert len(results) == 1
        assert results[0]["name"] == "nli"
        mock_export.assert_called_once()

    def test_export_all_embed_only(self):
        mock_result = {"name": "embed", "path": "/tmp/embed/model.onnx", "size_mb": 90.0}

        with patch(
            "agent_trust_lab.onnx_setup.export_model", return_value=mock_result
        ) as mock_export:
            results = export_all("embed")

        assert len(results) == 1
        assert results[0]["name"] == "embed"
        mock_export.assert_called_once()

    def test_export_all_invalid_filter(self):
        with pytest.raises(ValueError, match="Unknown model filter"):
            export_all("invalid")

    def test_export_all_custom_output_dir(self):
        mock_result = {"name": "nli", "path": "/custom/nli/model.onnx", "size_mb": 500.0}

        with patch(
            "agent_trust_lab.onnx_setup.export_model", return_value=mock_result
        ) as mock_export:
            results = export_all("nli", output_dir="/custom/nli")

        call_args = mock_export.call_args
        assert call_args[0][1] == "/custom/nli"
        assert results[0]["name"] == "nli"


class TestCLISetupOnnx:
    def test_setup_onnx_help(self):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["setup-onnx", "--help"])
        assert result.exit_code == 0
        assert "Export ONNX models" in result.stdout

    def test_setup_onnx_status(self):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["setup-onnx", "--status"])
        assert result.exit_code == 0
        assert "ONNX Model Status" in result.stdout

    def test_setup_onnx_invalid_model(self):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["setup-onnx", "--model", "invalid"])
        assert result.exit_code == 1
        assert "Invalid model" in result.stdout

    def test_setup_onnx_all_already_cached(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        cache = str(tmp_path / "onnx")
        for config in _EXPORT_CONFIGS.values():
            model_dir = os.path.join(cache, config.subdir)
            os.makedirs(model_dir, exist_ok=True)
            Path(os.path.join(model_dir, "model.onnx")).touch()

        monkeypatch.setattr("agent_trust_lab.onnx_setup.DEFAULT_CACHE", cache)

        runner = CliRunner()
        result = runner.invoke(app, ["setup-onnx", "--model", "all"])
        assert result.exit_code == 0
        assert "already cached" in result.stdout
