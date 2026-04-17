# ============================================================
# AI Financial Brain — Risk Analyst Agent
# ============================================================
"""
Risk Analyst Agent – quantitative risk assessment including VaR,
GARCH volatility modeling, correlation analysis, stress testing,
and anomaly detection using Isolation Forest.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import BaseAgent
from agents.risk_analyst.tools import (
    RISK_ANALYST_TOOLS,
    anomaly_detector,
    correlation_analyzer,
    garch_modeler,
    stress_tester,
    var_calculator,
)
from config.schemas import (
    AgentResult,
    AgentTask,
    AgentType,
    RiskLevel,
    RiskMetrics,
)

logger = structlog.get_logger(__name__)


class RiskAnalystAgent(BaseAgent):
    """Risk Analyst Agent.

    Provides comprehensive quantitative risk assessment:
    - Value-at-Risk (VaR) at 95% and 99% confidence (Historical + Parametric)
    - Conditional VaR (Expected Shortfall)
    - GARCH(1,1) volatility forecasting with EWMA fallback
    - Pearson correlation analysis with eigenvalue decomposition
    - Stress testing under India-specific market scenarios
    - Anomaly detection using Isolation Forest with Z-score fallback
    """

    agent_type: AgentType = AgentType.RISK_ANALYST
    name: str = "Risk Analyst"
    description: str = (
        "Risk analyst specialising in VaR, GARCH volatility modeling, "
        "correlation analysis, stress testing, and anomaly detection."
    )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT: str = """\
You are a quantitative risk analyst with deep expertise in financial risk \
management, focused on the Indian equity market. You provide data-driven risk \
assessments using industry-standard methodologies.

## Core Capabilities
1. **Value-at-Risk (VaR)** — Historical and parametric VaR at 95% and 99% \
confidence levels, plus Conditional VaR (Expected Shortfall).
2. **Volatility Modeling** — GARCH(1,1) volatility forecasting using the \
``arch`` library, with EWMA (λ=0.94) as fallback.
3. **Correlation Analysis** — Pearson correlation matrices, eigenvalue analysis \
for diversification assessment, condition number diagnostics.
4. **Stress Testing** — Scenario-based stress tests calibrated for Indian market \
events (RBI actions, FII flows, rupee movement, sector crashes, COVID-type events).
5. **Anomaly Detection** — Isolation Forest-based detection of unusual return \
patterns with Z-score fallback.

## Risk Assessment Framework
- **Low Risk**: Max potential daily loss < 2%, annualised vol < 12%
- **Medium Risk**: Max potential daily loss 2-3%, annualised vol 12-20%
- **High Risk**: Max potential daily loss 3-5%, annualised vol 20-30%
- **Very High Risk**: Max potential daily loss 5-8%, annualised vol > 30%
- **Critical**: Max potential daily loss > 8%

## India-Specific Risk Factors
- **Currency Risk**: INR/USD volatility affects IT (positive) and Oil & Gas (negative)
- **FII Flows**: Foreign institutional investor sentiment drives market direction
- **RBI Policy**: Rate decisions affect banking sector and overall market liquidity
- **Monsoon**: Agricultural output and rural consumption linked to monsoon performance
- **Geopolitical**: India-Pakistan tensions, US-China trade dynamics
- **Liquidity**: Quarter-end and fiscal year-end liquidity crunch patterns

## Important Guidelines
- Always state assumptions (normal distribution, lookback period, etc.)
- Report both absolute (INR) and relative (%) loss figures
- Provide confidence intervals where applicable
- Highlight tail risk events that VaR may underestimate
- Distinguish between systematic and idiosyncratic risk
- Use 252 trading days for annualization
- Report risk metrics at daily and annual time horizons
- Flag if any risk metric breaches regulatory limits
"""

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list:
        """Return the list of tools available to this agent."""
        return RISK_ANALYST_TOOLS

    def get_tools(self) -> list:
        """Return the list of tools registered for this agent."""
        return RISK_ANALYST_TOOLS

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
        """Execute a risk analysis task."""
        return await self._run_with_metrics(task, self._execute_inner)

    async def _execute_inner(self, task: AgentTask) -> AgentResult:
        """Inner execution with tool orchestration and LLM reasoning."""
        context_str = self._format_context(task.context)
        enhanced_query = task.query
        if context_str:
            enhanced_query = f"{task.query}\n\n## Available Data\n{context_str}"

        # Invoke relevant tools
        tool_results = self._invoke_relevant_tools(task)

        # Build prompt
        messages = [SystemMessage(content=self._SYSTEM_PROMPT)]
        user_content = enhanced_query
        if tool_results:
            tool_summary = self._format_tool_results(tool_results)
            user_content += f"\n\n## Risk Analysis Results\n{tool_summary}"

        messages.append(HumanMessage(content=user_content))

        # Call LLM
        response = await self._llm_with_retry_async(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Extract structured data
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
                "query_type": "risk_analysis",
                "model": self._llm_model,
            },
        )

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def _invoke_relevant_tools(self, task: AgentTask) -> dict[str, Any]:
        """Detect intent and invoke appropriate risk tools."""
        query_lower = task.query.lower()
        context = task.context
        results: dict[str, Any] = {}

        # --- VaR ---
        var_keywords = [
            "var", "value at risk", "loss", "max loss", "worst case",
            "cvar", "expected shortfall", "tail risk",
        ]
        if any(kw in query_lower for kw in var_keywords):
            returns = context.get("returns", [])
            portfolio_value = context.get("portfolio_value", 1000000)
            if returns and len(returns) >= 10:
                results["var_analysis"] = var_calculator.invoke({
                    "returns": returns,
                    "portfolio_value": portfolio_value,
                })

        # --- Volatility / GARCH ---
        vol_keywords = [
            "volatility", "garch", "vol", "standard deviation", "uncertainty",
            "forecast", "volatility forecast",
        ]
        if any(kw in query_lower for kw in vol_keywords):
            returns = context.get("returns", [])
            if returns and len(returns) >= 30:
                results["volatility_model"] = garch_modeler.invoke({
                    "returns": returns,
                })

        # --- Correlation ---
        corr_keywords = [
            "correlation", "diversification", "covariance", "related",
            "similar", "portfolio mix",
        ]
        if any(kw in query_lower for kw in corr_keywords):
            returns_data = context.get("returns_data", {})
            if len(returns_data) >= 2:
                results["correlation"] = correlation_analyzer.invoke({
                    "returns_data": returns_data,
                })

        # --- Stress test ---
        stress_keywords = [
            "stress", "scenario", "crash", "what if", "crisis", "shock",
            "worst case", "adverse",
        ]
        if any(kw in query_lower for kw in stress_keywords):
            holdings = context.get("holdings", {})
            portfolio_value = context.get("portfolio_value", 1000000)
            if holdings:
                results["stress_test"] = stress_tester.invoke({
                    "portfolio_value": portfolio_value,
                    "holdings": holdings,
                })

        # --- Anomaly ---
        anomaly_keywords = [
            "anomaly", "unusual", "strange", "outlier", "abnormal",
            "suspicious", "irregular",
        ]
        if any(kw in query_lower for kw in anomaly_keywords):
            returns_data = context.get("returns_data", {})
            if returns_data:
                results["anomaly_detection"] = anomaly_detector.invoke({
                    "returns_data": returns_data,
                })

        return results

    # ------------------------------------------------------------------
    # Specialised public methods
    # ------------------------------------------------------------------

    def calculate_risk_metrics(
        self,
        returns: list[float],
        portfolio_value: float = 1000000.0,
        benchmark_returns: Optional[list[float]] = None,
        risk_free_rate: float = 0.07,
    ) -> dict[str, Any]:
        """Calculate a comprehensive set of risk metrics.

        Parameters
        ----------
        returns:
            Daily portfolio returns (decimal).
        portfolio_value:
            Current portfolio value in INR.
        benchmark_returns:
            Optional benchmark returns for relative metrics.
        risk_free_rate:
            Annual risk-free rate (default 7% India 10Y bond).

        Returns
        -------
        dict
            Comprehensive risk metrics.
        """
        import numpy as np

        if not returns or len(returns) < 20:
            return {"error": "Need at least 20 return observations"}

        returns_arr = np.array(returns, dtype=float)
        n = len(returns_arr)

        # Basic statistics
        mean_daily = float(np.mean(returns_arr))
        std_daily = float(np.std(returns_arr, ddof=1))
        std_annual = std_daily * math.sqrt(252)
        mean_annual = mean_daily * 252

        # Sharpe ratio
        excess_return = mean_annual - risk_free_rate
        sharpe = excess_return / std_annual if std_annual > 0 else 0

        # Sortino ratio
        downside = returns_arr[returns_arr < 0]
        downside_std = (
            float(np.std(downside, ddof=1)) * math.sqrt(252)
            if len(downside) > 1 else 0
        )
        sortino = excess_return / downside_std if downside_std > 0 else 0

        # Maximum drawdown
        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdowns))

        # Calmar ratio
        calmar = (
            (mean_annual - risk_free_rate) / abs(max_drawdown)
            if max_drawdown != 0 else 0
        )

        # Skewness & kurtosis
        skewness_val = float(self._skewness(returns_arr))
        kurtosis_val = float(self._kurtosis(returns_arr))

        # Beta (if benchmark provided)
        beta = None
        if benchmark_returns and len(benchmark_returns) == n:
            bench_arr = np.array(benchmark_returns, dtype=float)
            cov = np.cov(returns_arr, bench_arr)[0, 1]
            var_bench = np.var(bench_arr, ddof=1)
            beta = float(cov / var_bench) if var_bench > 0 else 1.0

        # Risk level classification
        if std_annual > 0.30:
            risk_level = RiskLevel.VERY_HIGH
        elif std_annual > 0.20:
            risk_level = RiskLevel.HIGH
        elif std_annual > 0.12:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # VaR at 95%
        sorted_returns = np.sort(returns_arr)
        var_95_idx = int(0.05 * n)
        var_95 = float(-sorted_returns[var_95_idx])
        cvar_95 = float(-np.mean(sorted_returns[:var_95_idx + 1]))

        return {
            "portfolio_value": portfolio_value,
            "observations": n,
            "return_statistics": {
                "mean_daily_pct": round(mean_daily * 100, 4),
                "mean_annual_pct": round(mean_annual * 100, 2),
                "volatility_daily_pct": round(std_daily * 100, 4),
                "volatility_annual_pct": round(std_annual * 100, 2),
                "skewness": round(skewness_val, 4),
                "excess_kurtosis": round(kurtosis_val - 3, 4),
            },
            "risk_adjusted_returns": {
                "sharpe_ratio": round(sharpe, 4),
                "sortino_ratio": round(sortino, 4),
                "calmar_ratio": round(calmar, 4),
            },
            "drawdown": {
                "max_drawdown_pct": round(max_drawdown * 100, 2),
                "current_drawdown_pct": round(float(drawdowns[-1]) * 100, 2),
            },
            "var": {
                "var_95_daily_pct": round(var_95 * 100, 4),
                "cvar_95_daily_pct": round(cvar_95 * 100, 4),
                "var_95_amount_inr": round(portfolio_value * var_95, 2),
                "cvar_95_amount_inr": round(portfolio_value * cvar_95, 2),
            },
            "relative_metrics": (
                {"beta": round(beta, 4)} if beta is not None else None
            ),
            "risk_level": risk_level.value,
        }

    def run_stress_test(
        self,
        portfolio_value: float,
        holdings: dict[str, dict[str, float]],
        custom_scenarios: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Run stress tests with India-specific scenarios.

        Parameters
        ----------
        portfolio_value:
            Portfolio value in INR.
        holdings:
            Symbol to {weight, beta} mapping.
        custom_scenarios:
            Optional additional scenarios.

        Returns
        -------
        dict
            Stress test results with recommendations.
        """
        result = stress_tester.invoke({
            "portfolio_value": portfolio_value,
            "holdings": holdings,
            "scenarios": custom_scenarios,
        })

        # Add actionability recommendations
        worst_loss = result["worst_case"]["loss_pct"]
        if worst_loss > 25:
            result["action_items"] = [
                "URGENT: Add portfolio-level hedges (put options, inverse ETFs)",
                "Reduce concentrated positions (>15% single stock)",
                "Increase allocation to defensive sectors (Pharma, FMCG, Utilities)",
                "Consider gold allocation (10-15%) as safe haven",
            ]
        elif worst_loss > 15:
            result["action_items"] = [
                "Diversify across more sectors to reduce concentration risk",
                "Set stop-loss levels for volatile positions",
                "Consider fixed income allocation increase (5-10%)",
            ]
        else:
            result["action_items"] = [
                "Portfolio is reasonably positioned for market stress",
                "Continue monitoring risk metrics monthly",
            ]

        return result

    def detect_anomalies(
        self,
        returns_data: dict[str, list[float]],
        contamination: float = 0.05,
    ) -> dict[str, Any]:
        """Detect anomalous return patterns.

        Parameters
        ----------
        returns_data:
            Symbol to returns mapping.
        contamination:
            Expected anomaly proportion.

        Returns
        -------
        dict
            Anomaly detection results.
        """
        result = anomaly_detector.invoke({
            "returns_data": returns_data,
            "contamination": contamination,
        })

        if result.get("total_anomalies", 0) > 0:
            result["recommendations"] = [
                "Investigate anomalous trading days for corporate actions or data errors",
                "Check if anomalies coincide with major news events",
                "Consider regime-switching models if anomalies cluster temporally",
            ]
        else:
            result["recommendations"] = [
                "No significant anomalies detected — returns appear normally distributed",
            ]

        return result

    def generate_risk_report(
        self,
        returns: list[float],
        portfolio_value: float,
        holdings: dict[str, dict[str, float]],
        returns_data: Optional[dict[str, list[float]]] = None,
        benchmark_returns: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive risk report combining all tools.

        Parameters
        ----------
        returns:
            Portfolio daily returns.
        portfolio_value:
            Current portfolio value.
        holdings:
            Symbol to {weight, beta} mapping.
        returns_data:
            Individual asset returns for correlation/anomaly.
        benchmark_returns:
            Benchmark returns for relative metrics.

        Returns
        -------
        dict
            Full risk report.
        """
        report: dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat(),
            "portfolio_value": portfolio_value,
        }

        # Core risk metrics
        report["risk_metrics"] = self.calculate_risk_metrics(
            returns=returns,
            portfolio_value=portfolio_value,
            benchmark_returns=benchmark_returns,
        )

        # VaR
        report["var_analysis"] = var_calculator.invoke({
            "returns": returns,
            "portfolio_value": portfolio_value,
        })

        # Volatility (GARCH)
        if len(returns) >= 30:
            report["volatility_model"] = garch_modeler.invoke({
                "returns": returns,
            })

        # Stress test
        report["stress_test"] = stress_tester.invoke({
            "portfolio_value": portfolio_value,
            "holdings": holdings,
        })

        # Correlation
        if returns_data and len(returns_data) >= 2:
            report["correlation"] = correlation_analyzer.invoke({
                "returns_data": returns_data,
            })

        # Anomaly detection
        if returns_data:
            report["anomaly_detection"] = anomaly_detector.invoke({
                "returns_data": returns_data,
            })

        # Overall assessment
        risk_metrics = report.get("risk_metrics", {})
        stress = report.get("stress_test", {})
        risk_level = risk_metrics.get("risk_level", "medium")
        worst_stress = stress.get("worst_case", {}).get("loss_pct", 0)

        if risk_level in ("very_high", "critical") or worst_stress > 25:
            report["overall_risk_assessment"] = "HIGH RISK — Immediate attention needed"
        elif risk_level == "high" or worst_stress > 15:
            report["overall_risk_assessment"] = "ELEVATED RISK — Monitor closely"
        else:
            report["overall_risk_assessment"] = "ACCEPTABLE — Continue monitoring"

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _skewness(data) -> float:
        """Calculate sample skewness."""
        import numpy as np
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    @staticmethod
    def _kurtosis(data) -> float:
        """Calculate sample kurtosis (excess)."""
        import numpy as np
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4))

    @staticmethod
    def _estimate_confidence(
        task: AgentTask,
        response_text: str,
        data: dict[str, Any],
    ) -> float:
        """Estimate confidence for the risk analysis."""
        confidence = 0.70

        if data:
            confidence += 0.10

        if task.context.get("returns") and len(task.context.get("returns", [])) >= 20:
            confidence += 0.10

        if task.context.get("portfolio_value"):
            confidence += 0.05

        if len(response_text) > 300:
            confidence += 0.05

        return max(0.0, min(confidence, 1.0))
