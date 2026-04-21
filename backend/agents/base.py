# ============================================================
# AI Financial Brain — Base Agent Class
# ============================================================
"""
Abstract base class for all AI agents in the system.

Every agent inherits from :class:`BaseAgent`, which provides:
- Lazy LLM initialisation (Groq via LangChain client)
- Tenacity-based retry with exponential back-off
- Structured execution timing & metrics tracking
- Structured logging via ``structlog``
- A standard system-prompt builder
- Error handling that returns valid :class:`AgentResult` envelopes
"""

from __future__ import annotations

import abc
import functools
import json
import re
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Optional

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.config.settings import get_settings
from backend.config.schemas import (
    AgentResult,
    AgentTask,
    AgentType,
)
from backend.services.llm_service import generate_response

logger = structlog.get_logger(__name__)

# Retryable exception types for LLM / network calls
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class BaseAgent(abc.ABC):
    """Abstract base class for every AI agent in the system.

    Subclasses **must** set the three class-level descriptors and implement
    the three abstract methods.

    Example::

        class MyAgent(BaseAgent):
            agent_type = AgentType.MY_TYPE
            name = "My Agent"
            description = "Does something useful"

            @property
            def tools(self) -> list:
                return [tool_a, tool_b]

            async def execute(self, task: AgentTask) -> AgentResult:
                ...
    """

    # ------------------------------------------------------------------
    # Class-level descriptors (override in subclasses)
    # ------------------------------------------------------------------
    agent_type: AgentType
    name: str
    description: str

    def __init__(
        self,
        *,
        groq_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
        groq_temperature: float = 0.1,
        provider: str = "groq",
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialise the agent.

        Parameters
        ----------
        groq_api_key:
            Groq API key.
        groq_model:
            Model name used by the selected provider.
        groq_temperature:
            Sampling temperature (``0.0`` – ``2.0``).
        provider:
            ``"groq"``.
        verbose:
            If *True*, emit debug-level logs for every LLM call.
        **kwargs:
            Forwarded to subclasses.
        """
        settings = get_settings()
        selected_provider = str(settings.llm_provider or provider or "groq").lower().strip()
        self._groq_api_key: Optional[str] = groq_api_key or settings.groq_api_key
        self._llm_model: str = groq_model or settings.groq_model
        self._groq_temperature: float = groq_temperature
        self._provider: str = selected_provider
        self._verbose: bool = verbose
        self._llm: Optional[BaseChatModel] = None
        self._execution_count: int = 0
        self._total_execution_time_ms: float = 0.0
        self._error_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def llm(self) -> BaseChatModel:
        """Lazy-initialised LLM instance (cached after first call)."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    @property
    @abc.abstractmethod
    def tools(self) -> list:
        """Return the list of LangChain ``BaseTool`` instances for this agent."""

    @property
    def execution_count(self) -> int:
        """Total number of successful executions since startup."""
        return self._execution_count

    @property
    def total_execution_time_ms(self) -> float:
        """Cumulative wall-clock execution time in milliseconds."""
        return self._total_execution_time_ms

    @property
    def error_count(self) -> int:
        """Total number of errors encountered."""
        return self._error_count

    @property
    def avg_execution_time_ms(self) -> float:
        """Average execution time per call (ms)."""
        if self._execution_count == 0:
            return 0.0
        return self._total_execution_time_ms / self._execution_count

    # ------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute *task* and return a structured :class:`AgentResult`.

        This is the **primary entry point** called by the orchestrator.
        """

    @abc.abstractmethod
    def get_tools(self) -> list:
        """Return the list of tools registered for this agent."""

    @abc.abstractmethod
    def validate_input(self, task: AgentTask) -> bool:
        """Return *True* if *task* contains valid input for this agent."""

    # ------------------------------------------------------------------
    # LLM factory
    # ------------------------------------------------------------------

    def _create_llm(self) -> BaseChatModel:
        """Instantiate the language model based on the configured provider.

        Returns
        -------
        BaseChatModel
            A LangChain chat model ready for ``.invoke()`` / ``.ainvoke()``.
        """
        if self._provider != "groq":
            raise ValueError(
                f"Unsupported LLM_PROVIDER='{self._provider}'. "
                "This project is configured for Groq only. Set LLM_PROVIDER=groq."
            )

        return ChatGroq(
            groq_api_key=self._groq_api_key,
            model_name=self._llm_model,
            temperature=self._groq_temperature,
        )

    # ------------------------------------------------------------------
    # Retry decorator (exponential back-off, up to 3 attempts)
    # ------------------------------------------------------------------

    @staticmethod
    def _retry(
        max_attempts: int = 3,
        base_wait: float = 1.0,
        max_wait: float = 30.0,
        retry_types: tuple[type[Exception], ...] = _RETRYABLE_ERRORS,
    ) -> Callable:
        """Return a tenacity retry decorator.

        Parameters
        ----------
        max_attempts:
            Maximum number of retry attempts.
        base_wait:
            Base delay in seconds (exponential multiplier).
        max_wait:
            Maximum delay cap in seconds.
        retry_types:
            Tuple of exception types that trigger a retry.
        """
        return retry(
            retry=retry_if_exception_type(retry_types),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_wait, min=1, max=max_wait),
            reraise=True,
        )

    def _llm_with_retry(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> Any:
        """Invoke the LLM synchronously with tenacity retry wrapping.

        Parameters
        ----------
        messages:
            List of LangChain messages to send.
        **kwargs:
            Forwarded to ``llm.invoke()``.

        Returns
        -------
        The LLM response (``AIMessage``).
        """
        # Extract system and human messages for generate_response
        system_msg = next((m.content for m in messages if isinstance(m, SystemMessage)), "You are a financial AI assistant.")
        human_msg = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
        
        args = (human_msg, system_msg) if isinstance(system_msg, str) else (human_msg, "")

        @self._retry(max_attempts=3, retry_types=(Exception,))
        def _invoke() -> Any:
            # Wrap response in a mock AIMessage-like object if needed by caller
            from langchain_core.messages import AIMessage
            content = generate_response(*args)
            return AIMessage(content=content)

        return _invoke()

    async def _llm_with_retry_async(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> Any:
        """Async version of :meth:`_llm_with_retry`.

        Parameters
        ----------
        messages:
            List of LangChain messages to send.
        **kwargs:
            Forwarded to ``llm.ainvoke()``.

        Returns
        -------
        The LLM response (``AIMessage``).
        """
        # Extract system and human messages for generate_response
        system_msg = next((m.content for m in messages if isinstance(m, SystemMessage)), "You are a financial AI assistant.")
        human_msg = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
        
        args = (human_msg, system_msg) if isinstance(system_msg, str) else (human_msg, "")

        @self._retry(max_attempts=3, retry_types=(Exception,))
        async def _invoke() -> Any:
            from langchain_core.messages import AIMessage
            # generate_response is sync, so run in threadpool for async compatibility
            from starlette.concurrency import run_in_threadpool
            content = await run_in_threadpool(generate_response, *args)
            return AIMessage(content=content)

        return await _invoke()

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def _run_with_metrics(
        self, task: AgentTask, fn: Callable[..., Any]
    ) -> AgentResult:
        """Execute *fn* while tracking timing and error metrics.

        This wraps the inner business logic so every subclass gets
        consistent timing, logging, and error handling for free.

        Parameters
        ----------
        task:
            The incoming agent task.
        fn:
            An ``async`` callable ``(task) -> AgentResult``.

        Returns
        -------
        AgentResult
        """
        start = time.perf_counter()
        log = logger.bind(
            agent=self.name,
            agent_type=self.agent_type.value,
            task_id=str(task.task_id),
        )

        try:
            log.info("agent_execution_started")
            if not self.validate_input(task):
                log.warning("agent_input_validation_failed")
                elapsed_ms = (time.perf_counter() - start) * 1000
                return self._make_error_result(
                    task,
                    ValueError("Input validation failed"),
                    elapsed_ms=elapsed_ms,
                )

            result: AgentResult = await fn(task)

            elapsed_ms = (time.perf_counter() - start) * 1000
            # AgentResult.processing_time is set by the caller in fn,
            # but we record it here if they forgot.
            if result.processing_time == 0.0:
                result.processing_time = round(elapsed_ms / 1000, 4)

            self._execution_count += 1
            self._total_execution_time_ms += elapsed_ms

            log.info(
                "agent_execution_completed",
                elapsed_ms=round(elapsed_ms, 2),
                confidence=result.confidence,
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._error_count += 1
            log.error(
                "agent_execution_failed",
                error=str(exc),
                elapsed_ms=round(elapsed_ms, 2),
                traceback=traceback.format_exc(),
            )
            return self._make_error_result(task, exc, elapsed_ms=elapsed_ms)

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _make_success_result(
        self,
        task: AgentTask,
        output: str,
        confidence: float = 0.8,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """Build a successful :class:`AgentResult`.

        Parameters
        ----------
        task:
            The originating task.
        output:
            The primary output text / analysis.
        confidence:
            Self-assessed confidence (0-1).
        metadata:
            Optional structured metadata.

        Returns
        -------
        AgentResult
        """
        return AgentResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            output=output,
            confidence=confidence,
            processing_time=0.0,
            metadata=metadata or {},
        )

    def _make_error_result(
        self,
        task: AgentTask,
        error: Exception,
        elapsed_ms: float = 0.0,
    ) -> AgentResult:
        """Convert an exception into a structured :class:`AgentResult`.

        Parameters
        ----------
        task:
            The task that failed.
        error:
            The exception that was raised.
        elapsed_ms:
            Optional elapsed time to record.

        Returns
        -------
        AgentResult
            Always a valid result with error information in metadata.
        """
        return AgentResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            output=(
                f"Agent '{self.name}' encountered an error: "
                f"{type(error).__name__}: {error}"
            ),
            confidence=0.0,
            processing_time=round(elapsed_ms / 1000, 4),
            metadata={
                "error": True,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_system_prompt(
        *,
        role: str,
        instructions: str,
        constraints: str = "",
        context: str = "",
    ) -> str:
        """Build a standard system prompt with role, instructions, and constraints.

        Parameters
        ----------
        role:
            Who the agent is (e.g. "Senior Financial Advisor").
        instructions:
            Step-by-step instructions the agent must follow.
        constraints:
            Optional guardrails and limitations.
        context:
            Optional additional context block.

        Returns
        -------
        str
            A fully formed system prompt.
        """
        prompt = f"""You are {role}.

## Instructions
{instructions}
"""
        if constraints:
            prompt += f"""
## Constraints
{constraints}
"""
        if context:
            prompt += f"""
## Context
{context}
"""
        prompt += """
## Response Guidelines
- Be precise, data-driven, and actionable.
- When referencing financial figures, always include units (%, INR, Cr, etc.).
- If you are uncertain, state your confidence level explicitly.
- Structure responses with clear sections and bullet points where appropriate.
- For Indian markets, use INR unless otherwise specified.
- Follow SEBI regulations and disclose any assumptions made.
- Always include a disclaimer that this is not investment advice.
"""
        return prompt

    @staticmethod
    def _format_context(context: dict[str, Any], max_length: int = 1200) -> str:
        """Render a ``dict`` context into a readable block suitable for
        inclusion in a prompt.

        Long values are truncated to *max_length* total characters.

        Parameters
        ----------
        context:
            Arbitrary key-value pairs.
        max_length:
            Maximum total length of the rendered string.

        Returns
        -------
        str
        """
        if not context:
            return ""

        parts: list[str] = []
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                part = f"{key}: {json.dumps(value, default=str, ensure_ascii=False)}"
            else:
                part = f"{key}: {value}"
            parts.append(part)

        rendered = "\n".join(parts)
        if len(rendered) > max_length:
            rendered = rendered[:max_length] + "\n... [truncated]"
        return rendered

    # ------------------------------------------------------------------
    # Structured data extraction from LLM responses
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_from_text(text: str) -> dict[str, Any]:
        """Extract JSON blocks from LLM response text.

        Tries fenced code blocks first, then bare JSON objects.

        Parameters
        ----------
        text:
            Raw LLM text.

        Returns
        -------
        dict
            Extracted JSON, or empty dict if nothing found.
        """
        for pattern in [
            r"```json\s*\n(.*?)\n\s*```",
            r"```\s*\n(.*?)\n\s*```",
            r"\{[^{}]*\}",
        ]:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    continue
        return {}

    @staticmethod
    def _format_tool_results(tool_results: dict[str, Any]) -> str:
        """Format tool results into readable markdown for prompt injection.

        Parameters
        ----------
        tool_results:
            Tool name to result mapping.

        Returns
        -------
        str
        """
        lines: list[str] = []
        for tool_name, result in tool_results.items():
            if isinstance(result, dict) and "error" in result:
                lines.append(f"**{tool_name}**: Error — {result['error']}")
                continue
            lines.append(f"### {tool_name.replace('_', ' ').title()}")
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, dict):
                        lines.append(f"**{key}**:")
                        for k2, v2 in value.items():
                            lines.append(f"  - {k2}: {v2}")
                    elif isinstance(value, list):
                        preview = value[:5] if len(value) > 5 else value
                        lines.append(f"- {key}: {json.dumps(preview, default=str)}")
                    elif isinstance(value, (int, float, str, bool)):
                        lines.append(f"- {key}: {value}")
            lines.append("")
        return "\n".join(lines)

