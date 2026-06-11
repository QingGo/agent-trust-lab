import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_MODEL = "deepseek-v4-flash"

CACHE_ROOT = os.path.expanduser("~/.cache/agent-trust-lab")
ONNX_CACHE_DIR = os.path.join(CACHE_ROOT, "onnx")
CALIBRATION_CACHE_DIR = os.path.join(CACHE_ROOT, "calibration")
RESULT_CACHE_DIR = os.path.join(CACHE_ROOT, "results")


@dataclass
class EvaluationConfig:
    agent_type: str = "langchain"
    model: str = DEFAULT_MODEL
    judge_model: str = DEFAULT_MODEL
    api_key: str = ""
    base_url: str = ""
    sandbox: str = "docker"
    sandbox_image: str = "docker.m.daocloud.io/library/busybox:latest"
    sandbox_network: bool = False
    sandbox_tmpfs_size: str = "64m"
    docker_host: str = ""
    anchor_kb: str = "./kb/"
    trap_library_path: str = "./traps/"
    output_dir: str = "./output/results/"
    max_steps: int = 10
    parallel: int = 1
    calibration_profile: Optional[str] = None
    thinking_enabled: bool = False
    reasoning_effort: str = ""
    temperature: float = 0.0
    difficulty_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "trivial": 0.25,
            "easy": 0.5,
            "medium": 0.75,
            "hard": 1.0,
        }
    )
    gsar_vote_enabled: bool = True
    gsar_vote_models: List[str] = field(
        default_factory=lambda: ["deepseek-v4-flash", "deepseek-v4-pro"]
    )
    model_list: List[str] = field(default_factory=list)
    policy_rules: Optional[List[str]] = None
    codebase_path: Optional[str] = None
    test_suite_path: Optional[str] = None
    dry_run: bool = False
    skip_hallukg: bool = False
    strict_mode: bool = True
    timeout: int = 120
    skip_extract_types: List[str] = field(default_factory=lambda: ["action", "error"])
    grounded_threshold: float = 0.3
    nli_neutral_weight: float = 0.5
    anchor_type_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "semantic": 0.3,
            "token_overlap": 0.3,
            "multi_hop": 0.3,
            "none": 0.2,
        }
    )
    cache_enabled: bool = True
    cache_ttl_days: int = 7
    cache_dir: str = RESULT_CACHE_DIR
    adaptive_sampling: bool = True
    adaptive_disagreement_threshold: float = 0.3
    adaptive_max_samples: int = 3
    self_consistency_enabled: bool = False
    self_consistency_samples: int = 5
    injection_template: str = "system_note"
    runs: int = 1

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.parallel < 1:
            raise ValueError(f"parallel must be >= 1, got {self.parallel}")
        if self.timeout < 1:
            raise ValueError(f"timeout must be >= 1, got {self.timeout}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be in [0.0, 2.0], got {self.temperature}"
            )
        for diff, weight in self.difficulty_weights.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(
                    f"difficulty_weights[{diff}] must be in [0.0, 1.0], got {weight}"
                )
        if not 0.0 <= self.grounded_threshold <= 1.0:
            raise ValueError(
                f"grounded_threshold must be in [0.0, 1.0], got {self.grounded_threshold}"
            )
        if not 0.0 <= self.nli_neutral_weight <= 1.0:
            raise ValueError(
                f"nli_neutral_weight must be in [0.0, 1.0], got {self.nli_neutral_weight}"
            )
        for atype, weight in self.anchor_type_weights.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(
                    f"anchor_type_weights[{atype}] must be in [0.0, 1.0], got {weight}"
                )
        if self.cache_ttl_days < 0:
            raise ValueError(f"cache_ttl_days must be >= 0, got {self.cache_ttl_days}")
        if not 0.0 <= self.adaptive_disagreement_threshold <= 1.0:
            raise ValueError(
                f"adaptive_disagreement_threshold must be in [0.0, 1.0], "
                f"got {self.adaptive_disagreement_threshold}"
            )
        if self.adaptive_max_samples < 1:
            raise ValueError(
                f"adaptive_max_samples must be >= 1, got {self.adaptive_max_samples}"
            )
        if self.self_consistency_samples < 2:
            raise ValueError(
                f"self_consistency_samples must be >= 2, got {self.self_consistency_samples}"
            )
        if self.runs < 1:
            raise ValueError(f"runs must be >= 1, got {self.runs}")
        if self.gsar_vote_models and len(self.gsar_vote_models) < 2:
            raise ValueError(
                f"gsar_vote_models must have at least 2 models, got {self.gsar_vote_models}"
            )
