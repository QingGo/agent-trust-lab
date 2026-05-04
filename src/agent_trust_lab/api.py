"""High-level Python API for programmatic evaluation of agent reliability and hallucination.

Provides two convenience classes:

- TrustLab: General agent evaluation against the trap library.
- CodeLab: Code agent evaluation with code-specific checks.

These wrap the Orchestrator with sensible defaults and provide a simpler
interface for scripting and integration.
"""

from typing import Any, Dict, List, Optional

from agent_trust_lab.config import DEFAULT_MODEL, EvaluationConfig
from agent_trust_lab.orchestrator import Orchestrator


class TrustLab:
    """Python API for general agent trust evaluation.

    Usage:
        lab = TrustLab(model="deepseek-v4-flash")
        results = lab.run(trap_ids=["parameter_hallucination_01"])
        lab.report("results.json")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        agent_type: str = "langchain",
        base_url: str = "",
        api_key: str = "",
        sandbox: str = "docker",
        sandbox_image: str = "",
        sandbox_network: bool = False,
        docker_host: str = "",
        trap_library_path: str = "",
        parallel: int = 1,
        skip_hallukg: bool = False,
        timeout: int = 120,
        calibration_profile: Optional[str] = None,
    ):
        self.config = EvaluationConfig(
            model=model,
            agent_type=agent_type,
            base_url=base_url,
            api_key=api_key,
            sandbox=sandbox,
            sandbox_image=sandbox_image or "",
            sandbox_network=sandbox_network,
            docker_host=docker_host,
            trap_library_path=trap_library_path,
            parallel=parallel,
            skip_hallukg=skip_hallukg,
            timeout=timeout,
            calibration_profile=calibration_profile,
        )
        self._orchestrator: Optional[Orchestrator] = None

    @property
    def orchestrator(self) -> Orchestrator:
        if self._orchestrator is None:
            self._orchestrator = Orchestrator(self.config)
        return self._orchestrator

    def run(
        self,
        trap_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        mutate: bool = False,
        mutation_seed: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        """Run evaluation against selected traps.

        Args:
            trap_ids: Specific trap IDs to evaluate.
            category: Run all traps in a category (e.g., "general_agent").
            difficulty: Filter by difficulty (trivial, easy, medium, hard).
            mutate: Apply field and structural mutations.
            mutation_seed: Seed for reproducible mutations.
            limit: Max number of traps to run.

        Returns:
            List of EvaluationResult objects.
        """
        return self.orchestrator.run_traps(
            trap_ids=trap_ids,
            category=category,
            difficulty=difficulty,
            mutate=mutate,
            mutation_seed=mutation_seed,
            limit=limit,
        )

    def evaluate(self, trap_id: str):
        """Run a single trap evaluation and return the result.

        Args:
            trap_id: The trap ID to evaluate.

        Returns:
            EvaluationResult with compliance, hallucination, and code artifacts.
        """
        results = self.run(trap_ids=[trap_id])
        return results[0] if results else None

    def export(self, results, output_path: str) -> None:
        """Export evaluation results to JSON.

        Args:
            results: List of EvaluationResult from run().
            output_path: Path to write JSON file.
        """
        self.orchestrator.export_results(results, output_path)

    def report(self, json_path: str, output_path: Optional[str] = None) -> str:
        """Generate an HTML report from a JSON export file.

        Args:
            json_path: Path to JSON results file.
            output_path: Optional output HTML path.

        Returns:
            The HTML string.
        """
        from agent_trust_lab.report import ReportGenerator

        cal_data = None
        if self.config.calibration_profile:
            from agent_trust_lab.calibration.profile import load_profile

            profile = load_profile(self.config.calibration_profile)
            if profile:
                cal_data = profile.to_dict()

        generator = ReportGenerator()
        import json

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return generator.generate(data, output_path=output_path, calibration=cal_data)

    def list_traps(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List available traps with optional filtering.

        Args:
            category: Filter by category.
            difficulty: Filter by difficulty.

        Returns:
            List of trap summary dicts.
        """
        traps = self.orchestrator.trap_manager.load_traps(
            category=category,
            difficulty=difficulty,
        )
        return [
            {
                "trap_id": t.trap_id,
                "trap_type": t.trap_type,
                "severity": t.severity,
                "difficulty": t.difficulty,
                "category": t.category,
            }
            for t in traps
        ]


class CodeLab(TrustLab):
    """Python API for code agent trust evaluation.

    Extends TrustLab with code-specific defaults and code agent checks.

    Usage:
        lab = CodeLab(model="deepseek-v4-flash", codebase_path="./my-project")
        results = lab.run(trap_ids=["code_semantic_hallucination_01"])
        lab.export(results, "results.json")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        agent_type: str = "codex",
        codebase_path: Optional[str] = None,
        base_url: str = "",
        api_key: str = "",
        sandbox: str = "docker",
        sandbox_image: str = "",
        sandbox_network: bool = False,
        docker_host: str = "",
        trap_library_path: str = "",
        parallel: int = 1,
        skip_hallukg: bool = False,
        timeout: int = 120,
        calibration_profile: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            agent_type=agent_type,
            base_url=base_url,
            api_key=api_key,
            sandbox=sandbox,
            sandbox_image=sandbox_image,
            sandbox_network=sandbox_network,
            docker_host=docker_host,
            trap_library_path=trap_library_path,
            parallel=parallel,
            skip_hallukg=skip_hallukg,
            timeout=timeout,
            calibration_profile=calibration_profile,
        )
        self.config.codebase_path = codebase_path

    def run_code(
        self,
        trap_ids: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
        mutate: bool = False,
        mutation_seed: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        """Run code agent evaluation against traps in the 'code_agent' category.

        Args:
            trap_ids: Specific trap IDs to evaluate.
            difficulty: Filter by difficulty.
            mutate: Apply mutations.
            mutation_seed: Seed for mutations.
            limit: Max number of traps.

        Returns:
            List of EvaluationResult objects.
        """
        return self.run(
            trap_ids=trap_ids,
            category="code_agent",
            difficulty=difficulty,
            mutate=mutate,
            mutation_seed=mutation_seed,
            limit=limit,
        )
