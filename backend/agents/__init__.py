# ============================================================
# AI Financial Brain — Agent Package
# ============================================================
"""
Re-exports all six agents and provides a central agent registry.

Usage::

    from agents import AGENT_REGISTRY, PersonalCFOAgent, MarketAnalystAgent
"""

from __future__ import annotations

from config.schemas import AgentType

from agents.personal_cfo import PersonalCFOAgent
from agents.market_analyst import MarketAnalystAgent
from agents.news_sentiment import NewsSentimentAgent
from agents.risk_analyst import RiskAnalystAgent
from agents.portfolio_manager import PortfolioManagerAgent
from agents.critic import CriticAgent

# Import the base class for subclasses
from agents.base import BaseAgent

__all__ = [
    "BaseAgent",
    "PersonalCFOAgent",
    "MarketAnalystAgent",
    "NewsSentimentAgent",
    "RiskAnalystAgent",
    "PortfolioManagerAgent",
    "CriticAgent",
    "AGENT_REGISTRY",
]


# ============================================================
# Agent Registry — maps AgentType → Agent class
# ============================================================

AGENT_REGISTRY: dict[AgentType, type[BaseAgent]] = {
    AgentType.PERSONAL_CFO: PersonalCFOAgent,
    AgentType.MARKET_ANALYST: MarketAnalystAgent,
    AgentType.NEWS_SENTIMENT: NewsSentimentAgent,
    AgentType.RISK_ANALYST: RiskAnalystAgent,
    AgentType.PORTFOLIO_MANAGER: PortfolioManagerAgent,
    AgentType.CRITIC: CriticAgent,
}


def get_agent_class(agent_type: AgentType) -> type[BaseAgent]:
    """Look up the agent class for a given :class:`AgentType`.

    Parameters
    ----------
    agent_type:
        The agent type enum value.

    Returns
    -------
    type[BaseAgent]
        The corresponding agent class.

    Raises
    ------
    KeyError
        If the agent type is not registered.
    """
    agent_cls = AGENT_REGISTRY.get(agent_type)
    if agent_cls is None:
        raise KeyError(f"Unknown agent type: {agent_type}")
    return agent_cls


def create_agent(
    agent_type: AgentType,
    **kwargs,
) -> BaseAgent:
    """Factory function to instantiate an agent by type.

    Parameters
    ----------
    agent_type:
        The type of agent to create.
    **kwargs:
        Forwarded to the agent constructor.

    Returns
    -------
    BaseAgent
        An initialised agent instance.
    """
    agent_cls = get_agent_class(agent_type)
    return agent_cls(**kwargs)
