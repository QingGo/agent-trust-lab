from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvaluationConfig:
    agent_type: str = "langchain"
    model: str = "deepseek-v4-flash"
    api_key: str = ""
    base_url: str = ""
    sandbox: str = "docker"
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
    timeout: int = 120
