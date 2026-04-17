# ============================================================
# AI Financial Brain — Personal CFO Agent
# ============================================================
"""
Personal CFO Agent – provides personalised personal-finance guidance
including budgeting, savings-rate recommendations, insurance review,
tax-planning (India-specific), and goal-based financial planning.

Uses a RAG retrieval layer for Indian tax rules and personal-finance
knowledge, combined with structured tool execution for quantitative
analysis.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.base import BaseAgent
from backend.agents.personal_cfo.tools import (
    PERSONAL_CFO_TOOLS,
    budget_calculator,
    goal_planner,
    insurance_analyzer,
    tax_estimator,
)
from backend.config.schemas import (
    AgentResult,
    AgentTask,
    AgentType,
)

logger = structlog.get_logger(__name__)


class PersonalCFOAgent(BaseAgent):
    """Personal CFO Agent.

    Provides comprehensive personal finance guidance:
    - Budget analysis with 50/30/20 rule (Indian-adapted)
    - Tax planning under both Old & New regimes (AY 2025-26)
    - Goal-based financial planning with SIP calculators
    - Insurance needs analysis (term, health, life)
    - Retirement corpus planning
    """

    agent_type: AgentType = AgentType.PERSONAL_CFO
    name: str = "Personal CFO"
    description: str = (
        "Advises on budgeting, savings, insurance, tax planning, "
        "and retirement planning for Indian users."
    )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT: str = """\
You are a senior personal financial advisor specialising in Indian \
personal finance. You provide holistic financial guidance covering \
budgeting, tax optimisation, insurance, and goal-based investing.

## Core Capabilities
1. **Budget Analysis** — Evaluate spending patterns using the 50/30/20 rule \
adapted for Indian households (rent, EMIs, groceries, insurance, SIPs).
2. **Tax Planning** — Compare Old vs New tax regime, maximise deductions under \
Section 80C (₹1.5L), 80D (₹25K-50K), 80CCD(1B) (₹50K), HRA exemption, \
and home loan interest (Section 24b, ₹2L).
3. **Goal Planning** — Plan SIP amounts for retirement, child education, home \
purchase, and other goals using inflation-adjusted projections.
4. **Insurance Review** — Assess term life, health insurance, and critical \
illness cover based on income, dependents, and city tier.
5. **Retirement Planning** — Estimate retirement corpus using the 4% rule, \
annuity calculations, and NPS/EPF projections.

## Tax Rules (AY 2025-26)
- **New Regime**: Slabs: 0-4L (Nil), 4-8L (5%), 8-12L (10%), 12-16L (15%), \
16-20L (20%), 20-24L (25%), >24L (30%). Standard deduction: ₹75K. \
Rebate u/s 87A for income ≤ ₹12L.
- **Old Regime**: Slabs: 0-2.5L (Nil), 2.5-5L (5%), 5-10L (20%), >10L (30%). \
Standard deduction: ₹50K. Various deductions available.
- **LTCG on Equity**: 12.5% above ₹1.25L annual exemption (held > 12 months)
- **STCG on Equity**: 20% (held < 12 months)

## Important Guidelines
- Always mention whether the recommendation is for Old or New tax regime
- Use INR (₹) for all monetary amounts
- Consider Indian context: EMIs, SIPs, PPF, EPF, NPS, LIC, FD, RD
- Include specific section numbers for tax-saving instruments
- Account for 6% inflation in long-term projections
- Suggest specific instruments (PPF, ELSS, NPS Tier-I, Tax-saving FD)
- Flag if insurance coverage is inadequate
- For retirement: assume 4-6% withdrawal rate, 7% equity return, 6% debt return
"""

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list:
        """Return the list of tools available to this agent."""
        return PERSONAL_CFO_TOOLS

    def get_tools(self) -> list:
        """Return the list of tools registered for this agent."""
        return PERSONAL_CFO_TOOLS

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_input(self, task: AgentTask) -> bool:
        """Validate the task has usable content."""
        if not task.query or not task.query.strip():
            return False
        return True

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute a personal finance advisory task."""
        return await self._run_with_metrics(task, self._execute_inner)

    async def _execute_inner(self, task: AgentTask) -> AgentResult:
        """Inner execution with tool orchestration and LLM reasoning."""
        # Build enhanced query with context
        context_str = self._format_context(task.context)
        enhanced_query = task.query
        if context_str:
            enhanced_query = f"{task.query}\n\n## Available Context\n{context_str}"

        # Invoke relevant tools based on query intent
        tool_results = self._invoke_relevant_tools(task)

        # Build prompt with tool results
        messages = [SystemMessage(content=self._SYSTEM_PROMPT)]

        user_content = enhanced_query
        if tool_results:
            tool_summary = self._format_tool_results(tool_results)
            user_content += f"\n\n## Tool Analysis Results\n{tool_summary}"

        messages.append(HumanMessage(content=user_content))

        # Call LLM
        response = await self._llm_with_retry_async(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Extract any structured data from the response
        parsed_data = self._extract_json_from_text(response_text)
        if tool_results:
            parsed_data.update(tool_results)

        confidence = self._estimate_confidence(task, response_text, parsed_data)

        return AgentResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            output=response_text,
            confidence=confidence,
            processing_time=0.0,
            metadata={
                "tools_used": list(tool_results.keys()) if tool_results else [],
                "query_type": "personal_finance",
                "model": self._llm_model,
            },
        )

    # ------------------------------------------------------------------
    # Tool invocation based on intent detection
    # ------------------------------------------------------------------

    def _invoke_relevant_tools(self, task: AgentTask) -> dict[str, Any]:
        """Detect user intent and invoke appropriate tools."""
        query_lower = task.query.lower()
        context = task.context
        results: dict[str, Any] = {}

        # --- Budget analysis ---
        budget_keywords = [
            "budget", "spending", "expense", "saving rate", "50/30/20",
            "monthly budget", "cash flow", "how much should i spend",
        ]
        if any(kw in query_lower for kw in budget_keywords):
            monthly_income = context.get("monthly_income", 0)
            expenses = context.get("expenses", {})
            if monthly_income > 0 and expenses:
                savings_pct = context.get("savings_target_pct", 20.0)
                results["budget_analysis"] = budget_calculator.invoke({
                    "monthly_income": monthly_income,
                    "expenses": expenses,
                    "savings_target_pct": savings_pct,
                })

        # --- Tax estimation ---
        tax_keywords = [
            "tax", "income tax", "regime", "80c", "80d", "deduction",
            "tax saving", "old regime", "new regime", "slab",
            " itr", "tax return", "taxable",
        ]
        if any(kw in query_lower for kw in tax_keywords):
            annual_income = context.get("annual_income", 0)
            if annual_income > 0:
                results["tax_analysis"] = tax_estimator.invoke({
                    "annual_income": annual_income,
                    "regime": context.get("regime", "new"),
                    "deductions_80c": context.get("deductions_80c", 0),
                    "deductions_80d": context.get("deductions_80d", 0),
                    "deductions_80ccd": context.get("deductions_80ccd", 0),
                    "home_loan_interest": context.get("home_loan_interest", 0),
                    "hra_exempt": context.get("hra_exempt", 0),
                    "other_deductions": context.get("other_deductions", 0),
                })

        # --- Goal planning ---
        goal_keywords = [
            "goal", "plan", "retire", "child education", "home purchase",
            "car", "wedding", "sip", "target amount", "how much to invest",
            "corpus",
        ]
        if any(kw in query_lower for kw in goal_keywords):
            target = context.get("target_amount", 0)
            if target > 0:
                results["goal_plan"] = goal_planner.invoke({
                    "goal_name": context.get("goal_name", "Financial Goal"),
                    "target_amount": target,
                    "current_savings": context.get("current_savings", 0),
                    "monthly_contribution": context.get("monthly_contribution", 0),
                    "expected_return_pct": context.get("expected_return_pct", 12.0),
                    "time_horizon_years": context.get("time_horizon_years", 10.0),
                    "inflation_pct": context.get("inflation_pct", 6.0),
                })

        # --- Insurance analysis ---
        insurance_keywords = [
            "insurance", "insure", "cover", "term plan", "health insurance",
            "life cover", "premium", "sum assured", "protection",
        ]
        if any(kw in query_lower for kw in insurance_keywords):
            age = context.get("age", 30)
            annual_income = context.get("annual_income", 0)
            if age > 0 and annual_income > 0:
                results["insurance_analysis"] = insurance_analyzer.invoke({
                    "age": age,
                    "annual_income": annual_income,
                    "dependents": context.get("dependents", 0),
                    "existing_cover_life": context.get("existing_cover_life", 0),
                    "existing_cover_health": context.get("existing_cover_health", 0),
                    "existing_cover_term": context.get("existing_cover_term", 0),
                    "city_tier": context.get("city_tier", 1),
                })

        return results

    # ------------------------------------------------------------------
    # Specialised public methods
    # ------------------------------------------------------------------

    def analyze_financial_health(
        self,
        monthly_income: float,
        expenses: dict[str, float],
        annual_income: float = 0.0,
        age: int = 30,
        dependents: int = 0,
    ) -> dict[str, Any]:
        """Generate a comprehensive financial health assessment.

        Parameters
        ----------
        monthly_income:
            Gross monthly income in INR.
        expenses:
            Category-wise monthly expenses.
        annual_income:
            Annual gross income (auto-calculated if 0).
        age:
            User age.
        dependents:
            Number of dependents.

        Returns
        -------
        dict
            Combined analysis from budget, tax, insurance, and goal tools.
        """
        annual = annual_income or (monthly_income * 12)
        health: dict[str, Any] = {}

        # Budget
        health["budget"] = budget_calculator.invoke({
            "monthly_income": monthly_income,
            "expenses": expenses,
            "savings_target_pct": 20.0,
        })

        # Tax comparison
        health["tax"] = tax_estimator.invoke({
            "annual_income": annual,
            "regime": "new",
        })

        # Insurance check
        health["insurance"] = insurance_analyzer.invoke({
            "age": age,
            "annual_income": annual,
            "dependents": dependents,
        })

        # Emergency fund assessment
        total_expenses = sum(expenses.values())
        emergency_months = 6
        emergency_target = total_expenses * emergency_months
        health["emergency_fund"] = {
            "monthly_expenses": round(total_expenses, 2),
            "target_emergency_fund": round(emergency_target, 2),
            "months_covered": emergency_months,
        }

        return health

    def create_budget_plan(
        self,
        monthly_income: float,
        expenses: dict[str, float],
        savings_target_pct: float = 20.0,
    ) -> dict[str, Any]:
        """Create a detailed budget plan.

        Parameters
        ----------
        monthly_income:
            Gross monthly income in INR.
        expenses:
            Current expense breakdown.
        savings_target_pct:
            Desired savings rate.

        Returns
        -------
        dict
            Budget analysis with recommendations.
        """
        return budget_calculator.invoke({
            "monthly_income": monthly_income,
            "expenses": expenses,
            "savings_target_pct": savings_target_pct,
        })

    def suggest_tax_savings(
        self,
        annual_income: float,
        regime: str = "new",
        existing_80c: float = 0.0,
        existing_80d: float = 0.0,
        home_loan_interest: float = 0.0,
        hra_exempt: float = 0.0,
    ) -> dict[str, Any]:
        """Suggest tax-saving investments and regime choice.

        Parameters
        ----------
        annual_income:
            Annual income in INR.
        regime:
            Current regime preference.
        existing_80c:
            Current 80C investments.
        existing_80d:
            Current 80D investments.
        home_loan_interest:
            Annual home loan interest paid.
        hra_exempt:
            HRA exemption received.

        Returns
        -------
        dict
            Tax analysis with recommendations.
        """
        return tax_estimator.invoke({
            "annual_income": annual_income,
            "regime": regime,
            "deductions_80c": existing_80c,
            "deductions_80d": existing_80d,
            "home_loan_interest": home_loan_interest,
            "hra_exempt": hra_exempt,
        })

    def plan_retirement(
        self,
        current_age: int,
        retirement_age: int = 60,
        monthly_expenses_today: float = 30000.0,
        current_savings: float = 0.0,
        monthly_contribution: float = 10000.0,
        expected_return_pct: float = 12.0,
        inflation_pct: float = 6.0,
    ) -> dict[str, Any]:
        """Plan retirement corpus and monthly SIP targets.

        Parameters
        ----------
        current_age:
            Current age.
        retirement_age:
            Desired retirement age.
        monthly_expenses_today:
            Current monthly expenses in INR.
        current_savings:
            Current retirement savings in INR.
        monthly_contribution:
            Current monthly SIP for retirement.
        expected_return_pct:
            Expected annual return.
        inflation_pct:
            Expected inflation rate.

        Returns
        -------
        dict
            Retirement planning analysis.
        """
        years_to_retire = retirement_age - current_age
        if years_to_retire <= 0:
            return {"error": "Retirement age must be greater than current age"}

        # Future monthly expenses at retirement (inflation-adjusted)
        future_monthly = monthly_expenses_today * (1 + inflation_pct / 100) ** years_to_retire
        # Annual expenses at retirement
        future_annual = future_monthly * 12
        # Corpus needed (4% withdrawal rule, adjusted for Indian ~5% SWR)
        withdrawal_rate = 0.04
        required_corpus = future_annual / withdrawal_rate

        return goal_planner.invoke({
            "goal_name": f"Retirement at age {retirement_age}",
            "target_amount": required_corpus,
            "current_savings": current_savings,
            "monthly_contribution": monthly_contribution,
            "expected_return_pct": expected_return_pct,
            "time_horizon_years": float(years_to_retire),
            "inflation_pct": inflation_pct,
        })

    # ------------------------------------------------------------------
    # Confidence estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_confidence(
        task: AgentTask,
        response_text: str,
        data: dict[str, Any],
    ) -> float:
        """Estimate confidence for the personal finance analysis.

        Parameters
        ----------
        task:
            Original task.
        response_text:
            LLM output text.
        data:
            Extracted tool data.

        Returns
        -------
        float
            Confidence score between 0 and 1.
        """
        confidence = 0.70

        # More context = higher confidence
        if data:
            confidence += 0.10

        # If tools were invoked and returned results
        has_tool_results = any(
            isinstance(v, dict) and "error" not in v
            for v in data.values()
        )
        if has_tool_results:
            confidence += 0.10

        # Specific financial data in context
        if task.context.get("monthly_income") or task.context.get("annual_income"):
            confidence += 0.05

        # Response quality signals
        if len(response_text) > 300:
            confidence += 0.05

        uncertain_markers = ["insufficient information", "need more details"]
        if any(m in response_text.lower() for m in uncertain_markers):
            confidence -= 0.10

        return max(0.0, min(confidence, 1.0))

