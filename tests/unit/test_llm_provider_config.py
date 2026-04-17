from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure backend package imports like "from agents.base import BaseAgent" work in tests.
ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agents.base import BaseAgent  # noqa: E402
from config.schemas import AgentResult, AgentTask, AgentType  # noqa: E402
from services.agent_factory import _build_llm_from_settings  # noqa: E402


class _FakeChatGroq:
    def __init__(self, *, api_key: str | None = None, model: str | None = None, **kwargs):
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs


class _DummyAgent(BaseAgent):
    agent_type = AgentType.PERSONAL_CFO
    name = "DummyAgent"
    description = "Test agent"

    @property
    def tools(self) -> list:
        return []

    async def execute(self, task: AgentTask) -> AgentResult:
        raise NotImplementedError

    def get_tools(self) -> list:
        return []

    def validate_input(self, task: AgentTask) -> bool:
        return True


def test_base_agent_groq_provider_builds_chatgroq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    import agents.base as base_module

    monkeypatch.setattr(base_module, "ChatGroq", _FakeChatGroq)

    agent = _DummyAgent()
    llm = agent.llm

    assert isinstance(llm, _FakeChatGroq)
    assert llm.api_key == "test-groq-key"
    assert llm.model == "llama3-70b-8192"


def test_base_agent_non_groq_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    agent = _DummyAgent()

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        _ = agent.llm


def test_build_llm_from_settings_groq_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    import services.agent_factory as factory_module

    monkeypatch.setattr(factory_module, "ChatGroq", _FakeChatGroq)

    settings = SimpleNamespace(llm_provider="groq", groq_api_key="test-groq-key")
    llm = _build_llm_from_settings(settings)

    assert isinstance(llm, _FakeChatGroq)
    assert llm.api_key == "test-groq-key"
    assert llm.model == "llama3-70b-8192"


def test_build_llm_from_settings_non_groq_raises() -> None:
    settings = SimpleNamespace(llm_provider="openai", groq_api_key="test-groq-key")

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        _build_llm_from_settings(settings)
