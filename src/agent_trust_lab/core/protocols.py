"""Protocol definitions for dependency inversion.

These protocols define contracts for external dependencies (LLM APIs,
ML models, container runtimes). Modules depend on these abstractions
instead of concrete implementations, enabling:

1. Unit testing with mock implementations
2. Swapping implementations (e.g., a different LLM provider, a different
   NLI model, a different container runtime)
3. Clear documentation of what each dependency requires

Usage:
    from agent_trust_lab.core.protocols import LLMClient, NLIModel

    class MyClassifier:
        def __init__(self, llm: LLMClient):
            self._llm = llm

    # Production:
    classifier = MyClassifier(llm=DeepSeekClient())

    # Test:
    classifier = MyClassifier(llm=MockLLMClient())
"""

from typing import Any, Protocol


class _InstructorCompletions(Protocol):
    """Protocol for instructor's chat.completions.create() interface."""
    def create(self, **kwargs: Any) -> Any: ...


class _InstructorChat(Protocol):
    """Protocol for instructor's chat attribute with completions proxy."""
    @property
    def completions(self) -> "_InstructorCompletions": ...


class LLMClient(Protocol):
    """A configured instructor-wrapped LLM client for structured completions.

    Provides instructor-style structured output via Pydantic models:
        result = client.chat.completions.create(
            model=..., messages=..., response_model=SomePydanticModel, ...
        )

    Abstract over different LLM providers (DeepSeek, OpenAI, etc.) all
    accessed through the instructor interface.
    """

    @property
    def chat(self) -> "_InstructorChat": ...


class NLIModel(Protocol):
    """A Natural Language Inference model for faithfulness checking.

    Given a premise (evidence) and hypothesis (agent claim), returns an
    entailment/neutral/contradiction score.

    Abstract over ONNX deberta-base-mnli, HuggingFace pipeline, or any
    other NLI implementation.
    """

    def check(self, premise: str, hypothesis: str, neutral_weight: float = 0.5) -> float | None:
        """Score how well the hypothesis is supported by the premise.

        Args:
            premise: Evidence text.
            hypothesis: Claim text to evaluate.
            neutral_weight: Weight assigned to neutral class in [0, 1].
                Default 0.5 treats neutral as half-entailed.

        Returns:
            A score in [0, 1] where 1 = fully entailed, 0 = contradiction,
            or None if the model cannot process this input.
        """
        ...


class EmbeddingModel(Protocol):
    """A text embedding model for semantic similarity.

    Abstract over ONNX all-MiniLM-L6-v2, sentence-transformers, OpenAI
    embeddings API, or any other embedding provider.
    """

    def embed(self, text: str) -> list[float]:
        """Compute a dense vector embedding for the given text.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @property
    def is_available(self) -> bool:
        """Whether the embedding model is loaded and ready."""
        ...


class ContainerRuntime(Protocol):
    """A container/sandbox runtime for executing code safely.

    Abstract over Docker, Podman, or any other container engine.

    The runtime handles both container execution (run + wait + collect logs)
    and image lifecycle (pull, verify, cleanup).
    """

    def run(
        self,
        image: str,
        command: list[str],
        *,
        timeout: int = 30,
        network_enabled: bool = False,
        tmpfs_size: str = "64m",
        work_dir: str = "/tmp/sandbox",
        labels: dict[str, str] | None = None,
        read_only: bool = True,
        mem_limit: str = "128m",
        **kwargs: Any,
    ) -> tuple[int, str, str]:
        """Run a command inside a container and collect output.

        Blocks until the container exits or times out.

        Returns:
            A tuple of (exit_code, stdout, stderr).
        """
        ...

    def ensure_image(self, image_ref: str) -> bool:
        """Pull and verify a container image. Returns True if available."""
        ...

    def cleanup_orphaned(self, label: str) -> int:
        """Remove orphaned containers matching the given label.

        Returns count of containers removed.
        """
        ...
