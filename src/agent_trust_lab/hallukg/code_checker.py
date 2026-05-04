from typing import List, Optional

from agent_trust_lab.log import get_logger
from agent_trust_lab.models.report import CodeHalluReport
from agent_trust_lab.models.trajectory import SecureTrajectory

logger = get_logger("hallukg.code_checker")

_DEFAULT_PYTHON_IMAGE = "docker.m.daocloud.io/library/python:3-slim"

_ERROR_TYPE_MAP = {
    "ImportError": "mapping",
    "ModuleNotFoundError": "mapping",
    "AttributeError": "naming",
    "NameError": "naming",
    "TypeError": "parameter",
    "ValueError": "parameter",
    "SyntaxError": "logic_hallucination",
    "IndentationError": "logic_hallucination",
    "FileNotFoundError": "mapping",
    "OSError": "mapping",
}


class CodeHalluChecker:
    """Execute generated code in a sandbox to detect code hallucinations.

    Real execution: runs code in a Docker container (python:3-slim) and
    parses stderr for ImportError/AttributeError/TypeError etc.
    Falls back to stub on Docker unavailable or execution timeout.
    """

    def __init__(
        self,
        timeout: int = 30,
        docker_host: str = "",
        python_image: str = "",
    ):
        self.timeout = timeout
        self.docker_host = docker_host
        self.python_image = python_image or _DEFAULT_PYTHON_IMAGE

    def check(
        self,
        code: str,
        test_command: str = "",
        expected_error: Optional[str] = None,
        step_index: int = 0,
    ) -> CodeHalluReport:
        syntax_err = self._check_syntax(code)
        if syntax_err:
            return self._build_report(
                step_index=0,
                error_name="SyntaxError",
                error_msg=str(syntax_err),
                code_snippet=code[:100],
                expected_error=expected_error,
            )

        try:
            return self._check_with_docker(
                code, step_index=step_index, expected_error=expected_error
            )
        except Exception as e:
            logger.warning("CodeHalluChecker Docker execution failed, falling back to stub: %s", e)
            return self._stub_check(code, expected_error, step_index=step_index)

    def batch_check(self, trajectory: SecureTrajectory) -> List[CodeHalluReport]:
        code_steps = [(i, s) for i, s in enumerate(trajectory.steps) if s.type == "code_generation"]
        if not code_steps:
            return []

        reports: List[CodeHalluReport] = []
        for i, step in code_steps:
            try:
                report = self.check(code=step.content, step_index=i)
            except Exception as e:
                logger.warning(
                    "CodeHalluChecker real check failed for step %s, falling back to stub: %s",
                    i, e,
                )
                report = self._stub_check(step.content, step_index=i)
            reports.append(report)
        return reports

    def _check_syntax(self, code: str) -> Optional[SyntaxError]:
        try:
            compile(code, "<code_check>", "exec")
        except SyntaxError as e:
            return e
        return None

    def _check_with_docker(
        self,
        code: str,
        step_index: int = 0,
        expected_error: Optional[str] = None,
    ) -> CodeHalluReport:
        from agent_trust_lab.sandbox.image import (
            SANDBOX_LABEL,
            ImageManager,
            get_docker_client,
        )

        client = get_docker_client(self.docker_host)
        img_mgr = ImageManager(client)

        if not img_mgr.ensure_image(self.python_image):
            raise ValueError(f"Failed to pull Python image: {self.python_image}")

        container = client.containers.run(
            image=self.python_image,
            command=["python", "-c", code],
            detach=True,
            auto_remove=True,
            labels={SANDBOX_LABEL: ""},
            read_only=True,
            cap_drop=["ALL"],
            network_disabled=True,
            mem_limit="128m",
            nano_cpus=500_000_000,
        )

        try:
            result = container.wait(timeout=self.timeout)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            exit_code = result.get("StatusCode", -1)

            if exit_code == 0:
                return CodeHalluReport(
                    step_index=step_index,
                    hallucination_type="none",
                    code_snippet=code[:100],
                    error_message=None,
                    expected_error_pattern=expected_error,
                    fix_suggestion=None,
                )

            error_name, error_msg = self._parse_error(logs)
            return self._build_report(
                step_index=step_index,
                error_name=error_name or "RuntimeError",
                error_msg=error_msg or f"Exit code: {exit_code}",
                code_snippet=code[:100],
                expected_error=expected_error,
            )
        finally:
            try:
                container.remove(force=True)
            except Exception as e:
                logger.debug("Failed to remove code-check container: %s", e)

    def _parse_error(self, logs: str) -> tuple[Optional[str], Optional[str]]:
        import re

        pattern = r"(?P<name>[A-Za-z]\w*(?:Error|Warning|Exception))(?::\s*(?P<msg>.+))?"
        for line in logs.split("\n"):
            match = re.search(pattern, line.strip())
            if match:
                name = match.group("name")
                msg = match.group("msg") or line.strip()
                return name, msg
        return None, logs[:200].strip()

    def _classify_error(self, error_name: str, logs: str) -> str:
        if error_name in _ERROR_TYPE_MAP:
            return _ERROR_TYPE_MAP[error_name]
        return "logic_hallucination"

    def _build_report(
        self,
        step_index: int = 0,
        error_name: str = "",
        error_msg: str = "",
        code_snippet: str = "",
        expected_error: Optional[str] = None,
    ) -> CodeHalluReport:
        hallu_type = self._classify_error(error_name, error_msg)
        fix = self._suggest_fix(hallu_type, error_name, error_msg)
        return CodeHalluReport(
            step_index=step_index,
            hallucination_type=hallu_type,
            code_snippet=code_snippet,
            error_message=(
                f"{error_name}: {error_msg}"
                if error_name and error_msg
                else error_msg or error_name or None
            ),
            expected_error_pattern=expected_error,
            fix_suggestion=fix,
        )

    def _suggest_fix(self, hallu_type: str, error_name: str, error_msg: str) -> str:
        if hallu_type == "mapping":
            return "Verify the library/module name and its installation status."
        if hallu_type == "naming":
            return "Check function/class attribute names against API docs."
        if hallu_type == "parameter":
            return "Validate parameter names, types, and counts per function signature."
        return "Review code logic, imports, and variable definitions."

    def _stub_check(
        self,
        code: str,
        expected_error: Optional[str] = None,
        step_index: int = 0,
    ) -> CodeHalluReport:
        return CodeHalluReport(
            step_index=step_index,
            hallucination_type="naming",
            code_snippet=code[:100],
            error_message="Mock: ImportError for fake package (stub)",
            expected_error_pattern=expected_error,
            fix_suggestion="Use the correct package name from the verified list.",
        )
