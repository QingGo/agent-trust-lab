from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvaluationConfig:
    agent_type: str = "langchain"
    model: str = "deepseek-v4-flash"
    api_key: str = ""
    base_url: str = ""
    sandbox: str = "docker"
    sandbox_image: str = "docker.m.daocloud.io/library/busybox:latest"
    sandbox_network: bool = False
    sandbox_tmpfs_size: str = "64m"
    docker_host: str = ""
    anchor_kb: str = "./kb/"
    trap_library_path: str = "./traps/"
    output_dir: str = "./results/"
    max_steps: int = 10
    parallel: int = 1
    calibration_profile: Optional[str] = None
    model_list: List[str] = field(default_factory=list)
    policy_rules: Optional[List[str]] = None
    codebase_path: Optional[str] = None
    test_suite_path: Optional[str] = None
    dry_run: bool = False
    skip_hallukg: bool = False
    timeout: int = 120

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.parallel < 1:
            raise ValueError(f"parallel must be >= 1, got {self.parallel}")
        if self.timeout < 1:
            raise ValueError(f"timeout must be >= 1, got {self.timeout}")
