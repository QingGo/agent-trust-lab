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


class LLMClient(Protocol):
    """A configured instructor-wrapped LLM client for structured completions.

    Provides instructor-style structured output via Pydantic models:
        result = client.chat.completions.create(
            model=..., messages=..., response_model=SomePydanticModel, ...
        )

    Abstract over different LLM providers (DeepSeek, OpenAI, etc.) all
    accessed through the instructor interface.
    """

    class ChatProxy(Protocol):
        class CompletionsProxy(Protocol):
            def create(self, **kwargs: Any) -> Any: ...

        @property
        def completions(self) -> "LLMClient.CompletionsProxy": ...

    @property
    def chat(self) -> "LLMClient.ChatProxy": ...


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
    """

    def run(
        self,
        image: str,
        command: str,
        *,
        timeout: int = 30,
        network: bool = False,
        tmpfs_size: str = "64m",
        **kwargs: Any,
    ) -> tuple[int, str, str]:
        """Run a command inside a container.

        Returns:
            A tuple of (exit_code, stdout, stderr).
        """
        ...
