"""Personal CFO Agent — Tools.

LangChain-compatible tools for budgeting, tax estimation, goal planning,
and insurance analysis, all with an India-specific focus.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from langchain_core.tools import tool


# ============================================================
# Budget Calculator
# ============================================================

@tool
def budget_calculator(
    monthly_income: float,
    expenses: dict[str, float],
    savings_target_pct: float = 20.0,
) -> dict[str, Any]:
    """Calculate a recommended budget breakdown using the 50/30/20 rule
    adapted for Indian households (rent, EMIs, insurance, SIPs).

    Parameters
    ----------
    monthly_income:
        Gross monthly income in INR.
    expenses:
        Dictionary mapping expense categories to amounts, e.g.
        ``{"rent": 15000, "emi": 8000, "groceries": 5000}``.
    savings_target_pct:
        Target savings as a percentage of income (default 20%).

    Returns
    -------
    dict
        Budget allocation, surplus/deficit, and recommendations.
    """
    if monthly_income <= 0:
        return {"error": "Income must be positive"}

    total_expenses = sum(expenses.values())
    savings_target = monthly_income * (savings_target_pct / 100)
    surplus = monthly_income - total_expenses - savings_target

    # Recommended allocation using 50/30/20 for Indian context
    needs_budget = monthly_income * 0.50  # rent, EMIs, groceries, utilities
    wants_budget = monthly_income * 0.30  # dining, entertainment, shopping
    savings_budget = monthly_income * 0.20  # SIP, FD, PPF, emergency fund

    # Category-level analysis
    category_pcts: dict[str, float] = {}
    for category, amount in expenses.items():
        category_pcts[category] = round((amount / monthly_income) * 100, 1)

    # Identify high-spend categories (>30% of income)
    high_spend: list[str] = [
        cat for cat, pct in category_pcts.items() if pct > 30
    ]

    recommendations: list[str] = []
    if surplus < 0:
        recommendations.append(
            f"You are overspending by ₹{abs(surplus):,.0f}/month. "
            "Consider reducing discretionary expenses."
        )
    if high_spend:
        recommendations.append(
            f"High-spend alert: {', '.join(high_spend)} exceed 30% of income."
        )
    if total_expenses / monthly_income > 0.70:
        recommendations.append(
            "Your expense ratio is above 70%. Aim for ≤70% to maintain "
            "healthy savings."
        )
    if savings_target_pct < 20:
        recommendations.append(
            "Consider increasing savings to at least 20% for long-term "
            "financial security."
        )

    return {
        "monthly_income": monthly_income,
        "total_expenses": total_expenses,
        "expense_ratio": round(total_expenses / monthly_income * 100, 1),
        "savings_target": round(savings_target, 2),
        "surplus_deficit": round(surplus, 2),
        "recommended_allocation": {
            "needs_50pct": round(needs_budget, 2),
            "wants_30pct": round(wants_budget, 2),
            "savings_20pct": round(savings_budget, 2),
        },
        "category_breakdown": category_pcts,
        "high_spend_categories": high_spend,
        "recommendations": recommendations,
    }


# ============================================================
# Tax Estimator (India — Old & New Regime)
# ============================================================

@tool
def tax_estimator(
    annual_income: float,
    regime: str = "new",
    deductions_80c: float = 0.0,
    deductions_80d: float = 0.0,
    deductions_80ccd: float = 0.0,
    home_loan_interest: float = 0.0,
    hra_exempt: float = 0.0,
    other_deductions: float = 0.0,
) -> dict[str, Any]:
    """Estimate Indian income tax under both old and new regimes
    (AY 2025-26 slabs) and recommend the better option.

    Parameters
    ----------
    annual_income:
        Gross annual income in INR.
    regime:
        ``"old"`` or ``"new"`` — used as the primary calculation regime.
    deductions_80c:
        Section 80C deductions (PPF, ELSS, LIC, etc.), max ₹1.5L.
    deductions_80d:
        Section 80D deductions (Health Insurance), max ₹25K (₹50K for senior).
    deductions_80ccd:
        Section 80CCD(1B) — NPS additional, max ₹50K.
    home_loan_interest:
        Section 24(b) home loan interest, max ₹2L under old regime.
    hra_exempt:
        HRA exemption amount.
    other_deductions:
        Any other chapter-VI-A deductions.

    Returns
    -------
    dict
        Tax breakdown for both regimes, savings, and recommendation.
    """
    # --- New Regime Slabs (AY 2025-26) ---
    new_slabs = [
        (400000, 0.0),
        (80000, 0.05),
        (80000, 0.10),
        (80000, 0.15),
        (80000, 0.20),
        (80000, 0.25),
        (0, 0.30),  # remainder at 30%
    ]
    standard_deduction_new = 75000

    # --- Old Regime Slabs (AY 2025-26) ---
    old_slabs = [
        (250000, 0.0),
        (250000, 0.05),
        (500000, 0.10),
        (1000000, 0.20),
        (0, 0.30),
    ]
    standard_deduction_old = 50000

    def _calc_tax_old(income: float, ded_80c: float, ded_80d: float,
                      ded_80ccd: float, hl_interest: float, hra: float,
                      other: float) -> float:
        """Calculate tax under the old regime."""
        total_deductions = (
            min(ded_80c, 150000)
            + min(ded_80d, 25000)
            + min(ded_80ccd, 50000)
            + min(hl_interest, 200000)
            + hra
            + other
        )
        taxable = max(income - standard_deduction_old - total_deductions, 0)
        tax = 0.0
        remaining = taxable
        for slab_amount, rate in old_slabs:
            if remaining <= 0:
                break
            if slab_amount == 0:
                tax += remaining * rate
            else:
                chunk = min(remaining, slab_amount)
                tax += chunk * rate
                remaining -= chunk
        # Rebate u/s 87A for income ≤ 5L
        if taxable <= 500000:
            tax = max(tax - 12500, 0)
        # Add 4% Health & Education Cess
        tax *= 1.04
        # Surcharge for income > 50L
        if income > 5000000:
            surcharge_rate = 0.10 if income <= 10000000 else 0.15
            tax += tax * surcharge_rate
        return round(tax, 2)

    def _calc_tax_new(income: float) -> float:
        """Calculate tax under the new regime (no deductions except std)."""
        taxable = max(income - standard_deduction_new, 0)
        tax = 0.0
        remaining = taxable
        for slab_amount, rate in new_slabs:
            if remaining <= 0:
                break
            if slab_amount == 0:
                tax += remaining * rate
            else:
                chunk = min(remaining, slab_amount)
                tax += chunk * rate
                remaining -= chunk
        # Rebate u/s 87A for income ≤ 12L
        if taxable <= 1200000:
            tax = max(tax - 60000, 0)
        tax *= 1.04
        if income > 5000000:
            surcharge_rate = 0.10 if income <= 10000000 else 0.15
            tax += tax * surcharge_rate
        return round(tax, 2)

    tax_old = _calc_tax_old(
        annual_income, deductions_80c, deductions_80d,
        deductions_80ccd, home_loan_interest, hra_exempt, other_deductions,
    )
    tax_new = _calc_tax_new(annual_income)

    better_regime = "new" if tax_new <= tax_old else "old"
    tax_saving = abs(tax_old - tax_new)

    effective_rate_old = (
        round(tax_old / annual_income * 100, 2) if annual_income > 0 else 0
    )
    effective_rate_new = (
        round(tax_new / annual_income * 100, 2) if annual_income > 0 else 0
    )

    return {
        "annual_income": annual_income,
        "regime_used": regime,
        "old_regime": {
            "taxable_income": round(
                max(
                    annual_income
                    - standard_deduction_old
                    - min(deductions_80c, 150000)
                    - min(deductions_80d, 25000)
                    - min(deductions_80ccd, 50000)
                    - min(home_loan_interest, 200000)
                    - hra_exempt
                    - other_deductions,
                    0,
                ),
                2,
            ),
            "total_tax": tax_old,
            "effective_rate_pct": effective_rate_old,
        },
        "new_regime": {
            "taxable_income": round(
                max(annual_income - standard_deduction_new, 0), 2
            ),
            "total_tax": tax_new,
            "effective_rate_pct": effective_rate_new,
        },
        "recommended_regime": better_regime,
        "potential_saving": round(tax_saving, 2),
        "monthly_tax_old": round(tax_old / 12, 2),
        "monthly_tax_new": round(tax_new / 12, 2),
        "deductions_claimed": {
            "section_80c": min(deductions_80c, 150000),
            "section_80d": min(deductions_80d, 25000),
            "section_80ccd_1b": min(deductions_80ccd, 50000),
            "section_24b": min(home_loan_interest, 200000),
            "hra": hra_exempt,
            "other": other_deductions,
        },
    }


# ============================================================
# Goal Planner
# ============================================================

@tool
def goal_planner(
    goal_name: str,
    target_amount: float,
    current_savings: float = 0.0,
    monthly_contribution: float = 0.0,
    expected_return_pct: float = 12.0,
    time_horizon_years: float = 10.0,
    inflation_pct: float = 6.0,
) -> dict[str, Any]:
    """Plan savings to reach a financial goal, accounting for inflation
    and expected investment returns.

    Parameters
    ----------
    goal_name:
        Name of the goal, e.g. "Child's Education", "Retirement".
    target_amount:
        Desired corpus in INR (future value).
    current_savings:
        Already accumulated amount in INR.
    monthly_contribution:
        Planned monthly investment in INR.
    expected_return_pct:
        Expected annualised return (default 12% for equity-heavy).
    time_horizon_years:
        Years until the goal deadline.
    inflation_pct:
        Expected annual inflation (default 6% India).

    Returns
    -------
    dict
        Projected corpus, gap analysis, and SIP recommendations.
    """
    if time_horizon_years <= 0:
        return {"error": "Time horizon must be positive"}
    if target_amount <= 0:
        return {"error": "Target amount must be positive"}

    monthly_rate = expected_return_pct / 100 / 12
    inflation_factor = (1 + inflation_pct / 100) ** time_horizon_years
    real_target = target_amount  # assume target is already in future rupees

    # Future value of current savings (compound)
    fv_savings = current_savings * (1 + monthly_rate) ** (12 * time_horizon_years)

    # Future value of monthly contributions (SIP)
    if monthly_rate > 0:
        n = 12 * time_horizon_years
        fv_sip = monthly_contribution * (
            ((1 + monthly_rate) ** n - 1) / monthly_rate
        )
    else:
        fv_sip = monthly_contribution * 12 * time_horizon_years

    projected_corpus = fv_savings + fv_sip
    gap = real_target - projected_corpus

    # Required monthly SIP to close the gap
    required_monthly = 0.0
    if gap > 0 and monthly_rate > 0:
        n = 12 * time_horizon_years
        fv_factor = ((1 + monthly_rate) ** n - 1) / monthly_rate
        required_monthly = gap / fv_factor

    progress_pct = (
        min(projected_corpus / real_target * 100, 100) if real_target > 0 else 0
    )

    # Risk profile suggestion
    if time_horizon_years > 7:
        suggested_allocation = "Aggressive: 70% Equity, 20% Debt, 10% Gold"
    elif time_horizon_years > 3:
        suggested_allocation = "Moderate: 50% Equity, 35% Debt, 15% Gold"
    else:
        suggested_allocation = "Conservative: 20% Equity, 60% Debt, 20% Liquid"

    on_track = gap <= 0

    return {
        "goal_name": goal_name,
        "target_amount": round(real_target, 2),
        "time_horizon_years": time_horizon_years,
        "current_savings": round(current_savings, 2),
        "fv_of_savings": round(fv_savings, 2),
        "monthly_contribution": round(monthly_contribution, 2),
        "fv_of_sip": round(fv_sip, 2),
        "projected_corpus": round(projected_corpus, 2),
        "gap": round(gap, 2),
        "on_track": on_track,
        "progress_pct": round(progress_pct, 1),
        "required_monthly_sip": round(required_monthly, 2),
        "suggested_allocation": suggested_allocation,
        "inflation_adjusted_target": round(real_target / inflation_factor, 2),
    }


# ============================================================
# Insurance Analyzer
# ============================================================

@tool
def insurance_analyzer(
    age: int,
    annual_income: float,
    dependents: int = 0,
    existing_cover_life: float = 0.0,
    existing_cover_health: float = 0.0,
    existing_cover_term: int = 0,
    city_tier: int = 1,
) -> dict[str, Any]:
    """Analyse insurance needs and recommend adequate coverage for an
    Indian individual based on income, dependents, and city tier.

    Parameters
    ----------
    age:
        Current age of the individual.
    annual_income:
        Annual gross income in INR.
    dependents:
        Number of dependents (spouse, children, parents).
    existing_cover_life:
        Existing life insurance cover in INR.
    existing_cover_health:
        Existing health insurance cover in INR.
    existing_cover_term:
        Existing term insurance remaining term in years.
    city_tier:
        City tier (1 = Metro, 2 = Tier-2, 3 = Tier-3).

    Returns
    -------
    dict
        Coverage recommendations, gaps, and estimated premiums.
    """
    if age < 18 or age > 75:
        return {"error": "Age must be between 18 and 75"}

    # Life cover: 10-15x annual income (rule of thumb for India)
    life_cover_multiple = 15 if dependents > 0 else 10
    recommended_life_cover = annual_income * life_cover_multiple
    life_cover_gap = max(recommended_life_cover - existing_cover_life, 0)

    # Health cover: ₹10L base + ₹3L per dependent + city factor
    city_factor = {1: 1.5, 2: 1.2, 3: 1.0}.get(city_tier, 1.0)
    recommended_health_cover = (
        (1000000 + dependents * 300000) * city_factor
    )
    health_cover_gap = max(recommended_health_cover - existing_cover_health, 0)

    # Term insurance recommendation
    retirement_age = 60
    recommended_term = max(retirement_age - age, 10)

    # Estimated premiums (approximate for India market)
    term_premium_per_cr = {
        "25": 8000, "30": 10000, "35": 13000,
        "40": 18000, "45": 25000, "50": 35000,
    }
    age_bracket = str(min(max(age // 5 * 5, 25), 50))
    term_premium_per_cr_amount = term_premium_per_cr.get(age_bracket, 20000)
    estimated_term_premium = (
        round(life_cover_gap / 10000000 * term_premium_per_cr_amount, 2)
        if life_cover_gap > 0 else 0
    )

    # Health insurance premium (approx ₹15-20K per ₹10L cover)
    health_premium_rate = 18000 / 1000000
    estimated_health_premium = round(health_cover_gap * health_premium_rate, 2)

    total_annual_premium = estimated_term_premium + estimated_health_premium
    premium_as_income_pct = (
        round(total_annual_premium / annual_income * 100, 1)
        if annual_income > 0 else 0
    )

    # Emergency fund recommendation
    monthly_expenses_estimate = annual_income * 0.5 / 12
    emergency_fund = monthly_expenses_estimate * 6  # 6 months

    recommendations: list[str] = []
    if life_cover_gap > 0:
        recommendations.append(
            f"Get additional term life cover of ₹{life_cover_gap:,.0f} "
            f"(est. premium: ₹{estimated_term_premium:,.0f}/yr)"
        )
    if health_cover_gap > 0:
        recommendations.append(
            f"Increase health insurance to ₹{recommended_health_cover:,.0f} "
            f"(est. premium: ₹{estimated_health_premium:,.0f}/yr)"
        )
    if dependents > 0 and existing_cover_term < recommended_term:
        recommendations.append(
            f"Consider a {recommended_term}-year term plan to cover "
            f"dependents until retirement"
        )
    if premium_as_income_pct > 10:
        recommendations.append(
            "Total premium exceeds 10% of income — review coverage priorities"
        )

    return {
        "age": age,
        "annual_income": annual_income,
        "dependents": dependents,
        "city_tier": city_tier,
        "life_insurance": {
            "recommended_cover": round(recommended_life_cover, 2),
            "existing_cover": round(existing_cover_life, 2),
            "gap": round(life_cover_gap, 2),
            "recommended_term_years": recommended_term,
            "estimated_annual_premium": round(estimated_term_premium, 2),
        },
        "health_insurance": {
            "recommended_cover": round(recommended_health_cover, 2),
            "existing_cover": round(existing_cover_health, 2),
            "gap": round(health_cover_gap, 2),
            "estimated_annual_premium": round(estimated_health_premium, 2),
        },
        "total_annual_premium": round(total_annual_premium, 2),
        "premium_as_income_pct": premium_as_income_pct,
        "recommended_emergency_fund": round(emergency_fund, 2),
        "recommendations": recommendations,
    }


# ============================================================
# Export all tools as a list
# ============================================================

PERSONAL_CFO_TOOLS = [
    budget_calculator,
    tax_estimator,
    goal_planner,
    insurance_analyzer,
]
