"""Agent Factory - creates and manages agent instances."""
from __future__ import annotations

import importlib
import threading
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq

from backend.agents.base import BaseAgent
from backend.config.schemas import AgentType

logger = structlog.get_logger(__name__)

_AGENT_REGISTRY: dict[AgentType, tuple[str, str]] = {
    AgentType.PERSONAL_CFO: ("backend.agents.personal_cfo.agent", "PersonalCFOAgent"),
    AgentType.MARKET_ANALYST: ("backend.agents.market_analyst.agent", "MarketAnalystAgent"),
    AgentType.NEWS_SENTIMENT: ("backend.agents.news_sentiment.agent", "NewsSentimentAgent"),
    AgentType.RISK_ANALYST: ("backend.agents.risk_analyst.agent", "RiskAnalystAgent"),
    AgentType.PORTFOLIO_MANAGER: ("backend.agents.portfolio_manager.agent", "PortfolioManagerAgent"),
    AgentType.CRITIC: ("backend.agents.critic.agent", "CriticAgent"),
}

_AGENT_NAME_TO_TYPE: dict[str, AgentType] = {t.value: t for t in AgentType}


class AgentFactory:
    """Factory for creating and managing agent instances."""

    def __init__(self, llm: BaseChatModel | None = None, **agent_kwargs: Any) -> None:
        self._llm = llm
        self._agent_kwargs = agent_kwargs
        self._cache: dict[AgentType, BaseAgent] = {}
        self._lock = threading.Lock()
        self._overrides: dict[AgentType, type[BaseAgent]] = {}

    def create_agent(self, agent_type: AgentType | str) -> BaseAgent:
        """Return a BaseAgent subclass instance for *agent_type*."""
        if isinstance(agent_type, str):
            agent_type = _resolve_agent_type(agent_type)

        with self._lock:
            if agent_type in self._cache:
                return self._cache[agent_type]

            agent = self._instantiate(agent_type)
            self._cache[agent_type] = agent
            logger.info(
                "agent_factory.created",
                agent_type=agent_type.value,
                agent_class=type(agent).__name__,
            )
            return agent

    def get_all_agents(self) -> dict[AgentType, BaseAgent]:
        """Eagerly create and return all registered agents."""
        result: dict[AgentType, BaseAgent] = {}
        for agent_type in AgentType:
            result[agent_type] = self.create_agent(agent_type)
        return result

    def register_override(
        self, agent_type: AgentType | str, agent_class: type[BaseAgent]
    ) -> None:
        """Replace the default agent class for *agent_type* with a custom one."""
        if isinstance(agent_type, str):
            agent_type = _resolve_agent_type(agent_type)

        if not issubclass(agent_class, BaseAgent):
            raise TypeError(f"{agent_class} must be a subclass of BaseAgent")
        with self._lock:
            self._overrides[agent_type] = agent_class
            self._cache.pop(agent_type, None)
        logger.info("agent_factory.override_registered", agent_type=agent_type.value)

    def warm_cache(self) -> None:
        """Pre-instantiate every agent to avoid cold-start latency."""
        self.get_all_agents()

    def clear_cache(self) -> None:
        """Drop all cached agent instances."""
        with self._lock:
            self._cache.clear()
        logger.info("agent_factory.cache_cleared")

    def _instantiate(self, agent_type: AgentType) -> BaseAgent:
        """Create an agent instance for *agent_type*."""
        if agent_type in self._overrides:
            cls = self._overrides[agent_type]
            return cls(**self._agent_kwargs)

        module_path, class_name = _AGENT_REGISTRY[agent_type]
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"Cannot import agent {agent_type.value} from "
                f"{module_path}.{class_name}: {exc}"
            ) from exc

        if not issubclass(cls, BaseAgent):
            raise TypeError(
                f"{class_name} from {module_path} is not a BaseAgent subclass"
            )

        return cls(**self._agent_kwargs)

    @classmethod
    def from_settings(cls) -> "AgentFactory":
        """Build an AgentFactory using the application settings."""
        from backend.config.settings import get_settings

        settings = get_settings()
        llm = _build_llm_from_settings(settings)
        return cls(llm=llm)


def _resolve_agent_type(agent_type: str) -> AgentType:
    """Resolve a string like ``personal_cfo`` to an AgentType."""
    try:
        return AgentType(agent_type)
    except ValueError:
        raise ValueError(
            f"Unknown agent type '{agent_type}'. "
            f"Valid types: {[t.value for t in AgentType]}"
        )


def _build_llm_from_settings(settings: Any) -> BaseChatModel | None:
    """Instantiate a LangChain LLM from application settings using Groq."""
    provider = str(settings.llm_provider).lower().strip()

    if provider == "groq":
        return ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. "
        "This project is configured for Groq only. Set LLM_PROVIDER=groq."
    )

