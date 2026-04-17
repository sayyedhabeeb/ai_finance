"""
Tax rule ingestion pipeline for Indian tax regulations.

Ingests structured data (JSON, CSV, dict lists) of:
  - Income Tax Act sections (80C, 80D, 80CCD, 24(b), etc.)
  - LTCG / STCG rules
  - Tax slabs (old and new regimes)
  - Deduction limits
  - GST rules and rate categories

All data is chunked, embedded, and stored in the personal finance pgvector store.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

from backend.rag.embeddings.embedder import EmbeddingService
from backend.rag.models import RAGDocument
from backend.rag.personal_finance.pgvector_client import PersonalFinanceVectorStore

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data formatters — convert structured data to natural-language text
# ──────────────────────────────────────────────────────────────

def _format_tax_section(section: dict[str, Any]) -> str:
    """Format an IT Act section as readable text."""
    parts = [
        f"Section {section.get('section', 'N/A')}: {section.get('title', 'N/A')}",
        f"\nDescription: {section.get('description', 'N/A')}",
        f"\nCategory: {section.get('category', 'N/A')}",
        f"\nApplicability: {section.get('applicability', 'General taxpayers')}",
    ]

    limit = section.get("deduction_limit")
    if limit is not None:
        parts.append(f"\nDeduction Limit: ₹{limit:,.0f}")

    conditions = section.get("conditions")
    if conditions:
        parts.append(f"\nConditions: {json.dumps(conditions, indent=2) if isinstance(conditions, list) else conditions}")

    examples = section.get("examples")
    if examples:
        if isinstance(examples, list):
            parts.append("\nExamples:")
            for ex in examples:
                parts.append(f"  • {ex}")
        else:
            parts.append(f"\nExample: {examples}")

    notes = section.get("notes")
    if notes:
        parts.append(f"\nNotes: {notes}")

    return "\n".join(parts)


def _format_tax_slab(slab: dict[str, Any]) -> str:
    """Format a tax slab as readable text."""
    parts = [
        f"Tax Regime: {slab.get('regime', 'N/A')}",
        f"\nFinancial Year: {slab.get('fy', 'N/A')}",
        f"\nAssessment Year: {slab.get('ay', 'N/A')}",
        f"\nApplicable For: {slab.get('applicable_for', 'Individuals')}",
        f"\nTax Slabs:",
    ]

    slabs = slab.get("slabs", [])
    for s in slabs:
        lower = s.get("from", 0)
        upper = s.get("to")
        rate = s.get("rate", 0)
        cess = s.get("cess", "4%")
        if upper is not None:
            parts.append(f"  • ₹{lower:,.0f} - ₹{upper:,.0f}: {rate}% + {cess} Health & Education Cess")
        else:
            parts.append(f"  • Above ₹{lower:,.0f}: {rate}% + {cess} Health & Education Cess")

    rebate = slab.get("rebate")
    if rebate:
        parts.append(f"\nRebate under Section 87A: Tax rebate up to ₹{rebate.get('max_income', 0):,.0f} (max rebate ₹{rebate.get('max_rebate', 0):,.0f})")

    surcharge = slab.get("surcharge")
    if surcharge:
        parts.append(f"\nSurcharge: {json.dumps(surcharge, indent=2)}")

    return "\n".join(parts)


def _format_deduction(deduction: dict[str, Any]) -> str:
    """Format a deduction rule as readable text."""
    parts = [
        f"Deduction: {deduction.get('name', 'N/A')}",
        f"\nSection: {deduction.get('section', 'N/A')}",
        f"\nCategory: {deduction.get('category', 'N/A')}",
        f"\nMaximum Limit: ₹{deduction.get('max_limit', 0):,.0f}",
        f"\nDescription: {deduction.get('description', 'N/A')}",
    ]

    eligible = deduction.get("eligible_investments")
    if eligible:
        parts.append(f"\nEligible Investments/Instruments:")
        if isinstance(eligible, list):
            for inv in eligible:
                parts.append(f"  • {inv}")
        else:
            parts.append(f"  • {eligible}")

    conditions = deduction.get("conditions")
    if conditions:
        parts.append(f"\nConditions:")
        if isinstance(conditions, list):
            for c in conditions:
                parts.append(f"  • {c}")
        else:
            parts.append(f"  • {conditions}")

    return "\n".join(parts)


def _format_gst_rule(rule: dict[str, Any]) -> str:
    """Format a GST rule as readable text."""
    parts = [
        f"GST Rule: {rule.get('name', 'N/A')}",
        f"\nCategory: {rule.get('category', 'N/A')}",
        f"\nGST Rate: {rule.get('rate', 'N/A')}%",
        f"\nHSN/SAC Code: {rule.get('hsn_code', 'N/A')}",
        f"\nDescription: {rule.get('description', 'N/A')}",
    ]

    exemptions = rule.get("exemptions")
    if exemptions:
        parts.append(f"\nExemptions:")
        if isinstance(exemptions, list):
            for ex in exemptions:
                parts.append(f"  • {ex}")
        else:
            parts.append(f"  • {exemptions}")

    itc_available = rule.get("itc_available", False)
    parts.append(f"\nInput Tax Credit (ITC) Available: {'Yes' if itc_available else 'No'}")

    return "\n".join(parts)


def _format_capital_gain(rule: dict[str, Any]) -> str:
    """Format a capital gains rule (LTCG/STCG)."""
    parts = [
        f"{rule.get('type', 'Capital Gain')} Rules",
        f"\nAsset Class: {rule.get('asset_class', 'N/A')}",
        f"\nHolding Period: {rule.get('holding_period', 'N/A')}",
        f"\nTax Rate: {rule.get('tax_rate', 'N/A')}%",
        f"\nDescription: {rule.get('description', 'N/A')}",
    ]

    exemption = rule.get("exemption_limit")
    if exemption is not None:
        parts.append(f"\nExemption Limit: ₹{exemption:,.0f}")

    indexation = rule.get("indexation_available", False)
    parts.append(f"\nIndexation Benefit: {'Available' if indexation else 'Not Available'}")

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────
# Main ingestion pipeline
# ──────────────────────────────────────────────────────────────

class TaxRuleIngestionPipeline:
    """Ingests Indian tax rules, sections, and regulations into pgvector.

    Parameters
    ----------
    vector_store:
        Connected :class:`PersonalFinanceVectorStore`.
    embedder:
        :class:`EmbeddingService` for computing embeddings.
    jurisdiction:
        Default jurisdiction code (``"IN"`` for India).
    """

    def __init__(
        self,
        vector_store: PersonalFinanceVectorStore,
        embedder: EmbeddingService,
        jurisdiction: str = "IN",
    ) -> None:
        self._vs = vector_store
        self._embedder = embedder
        self._jurisdiction = jurisdiction

    # ── Public API ──────────────────────────────────────────

    async def ingest_tax_act(self, sections: list[dict[str, Any]]) -> int:
        """Ingest Income Tax Act sections.

        Parameters
        ----------
        sections:
            List of section dicts.  Each should have at least:
            ``section``, ``title``, ``description``.  Optional:
            ``category``, ``deduction_limit``, ``conditions``, ``examples``, ``notes``.

        Returns
        -------
        int
            Number of chunks stored.
        """
        docs = []
        for sec in sections:
            text = _format_tax_section(sec)
            doc = self._build_doc(
                content=text,
                title=f"Section {sec.get('section', 'N/A')}: {sec.get('title', '')}",
                doc_type="tax_section",
                category=sec.get("category", "tax_act"),
                source=sec.get("source", "income_tax_act"),
                metadata={"section": sec.get("section"), "raw_data": sec},
            )
            docs.append(doc)
        return await self._ingest_docs(docs)

    async def ingest_deductions(self, deductions: list[dict[str, Any]]) -> int:
        """Ingest deduction rules (80C, 80D, 80CCD, 24(b), etc.).

        Parameters
        ----------
        deductions:
            List of deduction dicts with: ``name``, ``section``, ``category``,
            ``max_limit``, ``description``, ``eligible_investments``, ``conditions``.

        Returns
        -------
        int
            Number of chunks stored.
        """
        docs = []
        for ded in deductions:
            text = _format_deduction(ded)
            doc = self._build_doc(
                content=text,
                title=f"Deduction {ded.get('section', '')}: {ded.get('name', '')}",
                doc_type="deduction_rule",
                category="deductions",
                source=ded.get("source", "income_tax_act"),
                metadata={"section": ded.get("section"), "max_limit": ded.get("max_limit"), "raw_data": ded},
            )
            docs.append(doc)
        return await self._ingest_docs(docs)

    async def ingest_slabs(self, tax_slabs: list[dict[str, Any]]) -> int:
        """Ingest tax slab structures (old and new regimes).

        Parameters
        ----------
        tax_slabs:
            List of slab dicts with: ``regime``, ``fy``, ``ay``,
            ``applicable_for``, ``slabs`` (list of ``{from, to, rate, cess}``),
            ``rebate``, ``surcharge``.

        Returns
        -------
        int
            Number of chunks stored.
        """
        docs = []
        for slab in tax_slabs:
            text = _format_tax_slab(slab)
            doc = self._build_doc(
                content=text,
                title=f"Tax Slabs - {slab.get('regime', '')} Regime - FY {slab.get('fy', '')}",
                doc_type="tax_slab",
                category="tax_slabs",
                source=slab.get("source", "income_tax_act"),
                metadata={"regime": slab.get("regime"), "fy": slab.get("fy"), "raw_data": slab},
            )
            docs.append(doc)
        return await self._ingest_docs(docs)

    async def ingest_gst_rules(self, rules: list[dict[str, Any]]) -> int:
        """Ingest GST rules and rate categories.

        Parameters
        ----------
        rules:
            List of GST rule dicts with: ``name``, ``category``, ``rate``,
            ``hsn_code``, ``description``, ``exemptions``, ``itc_available``.

        Returns
        -------
        int
            Number of chunks stored.
        """
        docs = []
        for rule in rules:
            text = _format_gst_rule(rule)
            doc = self._build_doc(
                content=text,
                title=f"GST: {rule.get('name', '')} ({rule.get('rate', '')}%)",
                doc_type="gst_rule",
                category="gst",
                source=rule.get("source", "gst_act"),
                metadata={"hsn_code": rule.get("hsn_code"), "rate": rule.get("rate"), "raw_data": rule},
            )
            docs.append(doc)
        return await self._ingest_docs(docs)

    async def ingest_capital_gains(self, rules: list[dict[str, Any]]) -> int:
        """Ingest LTCG/STCG rules.

        Parameters
        ----------
        rules:
            List of capital gain dicts with: ``type``, ``asset_class``,
            ``holding_period``, ``tax_rate``, ``description``,
            ``exemption_limit``, ``indexation_available``.

        Returns
        -------
        int
            Number of chunks stored.
        """
        docs = []
        for rule in rules:
            text = _format_capital_gain(rule)
            doc = self._build_doc(
                content=text,
                title=f"{rule.get('type', 'Capital Gain')}: {rule.get('asset_class', '')}",
                doc_type="tax_rule",
                category="capital_gains",
                source=rule.get("source", "income_tax_act"),
                metadata={
                    "type": rule.get("type"),
                    "asset_class": rule.get("asset_class"),
                    "raw_data": rule,
                },
            )
            docs.append(doc)
        return await self._ingest_docs(docs)

    async def ingest_from_json_file(self, file_path: str) -> int:
        """Ingest tax data from a JSON file.

        The JSON file should be an object with keys like
        ``"sections"``, ``"deductions"``, ``"slabs"``, ``"gst_rules"``,
        ``"capital_gains"``.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = 0
        if "sections" in data:
            total += await self.ingest_tax_act(data["sections"])
        if "deductions" in data:
            total += await self.ingest_deductions(data["deductions"])
        if "slabs" in data:
            total += await self.ingest_slabs(data["slabs"])
        if "gst_rules" in data:
            total += await self.ingest_gst_rules(data["gst_rules"])
        if "capital_gains" in data:
            total += await self.ingest_capital_gains(data["capital_gains"])

        logger.info("Ingested %d total chunks from JSON file: %s", total, file_path)
        return total

    async def ingest_from_csv_file(self, file_path: str, data_type: str = "sections") -> int:
        """Ingest tax data from a CSV file.

        Parameters
        ----------
        file_path:
            Path to the CSV file.
        data_type:
            One of ``"sections"``, ``"deductions"``, ``"slabs"``, ``"gst_rules"``.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        dispatch = {
            "sections": self.ingest_tax_act,
            "deductions": self.ingest_deductions,
            "slabs": self.ingest_slabs,
            "gst_rules": self.ingest_gst_rules,
        }

        ingest_fn = dispatch.get(data_type)
        if ingest_fn is None:
            raise ValueError(f"Unknown data_type: {data_type!r}. Choose from {list(dispatch.keys())}")

        return await ingest_fn(rows)

    # ── Internal ────────────────────────────────────────────

    def _build_doc(
        self,
        content: str,
        title: str,
        doc_type: str,
        category: str,
        source: str,
        metadata: dict[str, Any],
    ) -> RAGDocument:
        """Build a :class:`RAGDocument` for a tax rule."""
        return RAGDocument(
            content=content,
            title=title,
            source=source,
            doc_type=doc_type,  # type: ignore[arg-type]
            category=category,
            jurisdiction=self._jurisdiction,
            metadata=metadata,
        )

    async def _ingest_docs(self, docs: list[RAGDocument]) -> int:
        """Chunk, embed, and store a list of documents."""
        if not docs:
            return 0

        from backend.rag.chunking.strategies import DocumentChunker

        chunker = DocumentChunker()
        all_ingest_dicts: list[dict[str, Any]] = []

        for doc in docs:
            # Tax rules are usually short enough for a single chunk,
            # but handle longer ones via recursive chunking
            chunks = chunker.recursive_chunk(doc.content, chunk_size=1500, overlap=100)

            # Embed all chunks
            embeddings = self._embedder.embed_batch(chunks)

            chunk_list: list[dict[str, Any]] = []
            for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
                chunk_list.append({
                    "content": chunk_text,
                    "embedding": emb,
                    "metadata": {
                        **doc.metadata,
                        "chunk_index": idx,
                        "doc_title": doc.title,
                        "doc_type": doc.doc_type.value,
                    },
                })

            all_ingest_dicts.append({
                "content": doc.content,
                "doc_type": doc.doc_type.value,
                "category": doc.category,
                "jurisdiction": doc.jurisdiction,
                "source": doc.source,
                "title": doc.title,
                "checksum": doc.checksum,
                "chunks": chunk_list,
            })

        return await self._vs.batch_ingest(all_ingest_dicts)

    # ── Sub-question decomposition helpers ──────────────────

    @staticmethod
    def get_sample_india_tax_data() -> dict[str, list[dict[str, Any]]]:
        """Return sample Indian tax data for testing / initial seeding.

        Covers:
          - Key IT Act sections (80C, 80D, 80CCD(1B), 24(b))
          - Old and New regime tax slabs for FY 2024-25
          - LTCG/STCG rules for equity and debt
          - Key GST rate categories
          - Common deductions
        """
        return {
            "sections": [
                {
                    "section": "80C",
                    "title": "Deduction for Investments in Specified Instruments",
                    "description": "Allows deduction of up to ₹1.5 lakh from gross total income for investments in specified instruments like PPF, ELSS, NSC, Life Insurance Premium, 5-year Fixed Deposits, NPS Tier-1, etc.",
                    "category": "deductions",
                    "deduction_limit": 150000,
                    "applicability": "Individuals and HUFs",
                    "conditions": [
                        "Investment must be made during the financial year",
                        "Lock-in period varies by instrument (3-15 years)",
                        "Deduction available under old tax regime only",
                    ],
                    "examples": [
                        "If you invest ₹1,50,000 in PPF, you can deduct the entire amount from taxable income under Section 80C.",
                        "Combination: ₹50,000 in ELSS + ₹1,00,000 in PPF = ₹1,50,000 deduction.",
                    ],
                    "notes": "Not available under the new tax regime (Section 115BAC).",
                },
                {
                    "section": "80D",
                    "title": "Deduction for Health Insurance Premium",
                    "description": "Deduction for health insurance premium paid for self, family, and parents. Available under both old and new tax regimes.",
                    "category": "deductions",
                    "deduction_limit": 75000,
                    "applicability": "Individuals and HUFs",
                    "conditions": [
                        "Self, spouse, and children: up to ₹25,000 (₹50,000 for senior citizens)",
                        "Parents: up to ₹25,000 (₹50,000 if parents are senior citizens)",
                        "Preventive health check-up: ₹5,000 within the overall limit",
                    ],
                    "examples": [
                        "Self (non-senior) + Parents (senior): ₹25,000 + ₹50,000 = ₹75,000 deduction.",
                    ],
                    "notes": "Available under both old and new tax regimes.",
                },
                {
                    "section": "80CCD(1B)",
                    "title": "Additional Deduction for NPS Contributions",
                    "description": "Additional deduction of up to ₹50,000 for contributions to the National Pension System (NPS) Tier-1 account, over and above the ₹1.5 lakh limit under Section 80C.",
                    "category": "deductions",
                    "deduction_limit": 50000,
                    "applicability": "Individuals (salaried and self-employed)",
                    "conditions": [
                        "Contribution must be to NPS Tier-1 account",
                        "Available under both old and new tax regimes",
                    ],
                    "examples": [
                        "If you contribute ₹50,000 to NPS, you get an additional ₹50,000 deduction beyond Section 80C.",
                    ],
                },
                {
                    "section": "24(b)",
                    "title": "Deduction for Interest on Home Loan",
                    "description": "Deduction for interest paid on home loan for self-occupied property. Maximum deduction ₹2,00,000 for self-occupied and no limit for let-out property.",
                    "category": "deductions",
                    "deduction_limit": 200000,
                    "applicability": "Individuals and HUFs with home loans",
                    "conditions": [
                        "Loan must be for acquisition/construction of residential property",
                        "Construction must be completed within 5 years of loan sanction",
                        "For self-occupied property: max ₹2,00,000 (₹30,000 if construction not completed in 5 years)",
                        "For let-out property: full interest is deductible",
                    ],
                    "examples": [
                        "Home loan interest of ₹2,50,000 on self-occupied property: deduction = ₹2,00,000.",
                    ],
                },
            ],
            "slabs": [
                {
                    "regime": "Old",
                    "fy": "2024-25",
                    "ay": "2025-26",
                    "applicable_for": "Individuals (< 60 years)",
                    "slabs": [
                        {"from": 0, "to": 250000, "rate": 0, "cess": "4%"},
                        {"from": 250000, "to": 500000, "rate": 5, "cess": "4%"},
                        {"from": 500000, "to": 1000000, "rate": 20, "cess": "4%"},
                        {"from": 1000000, "to": None, "rate": 30, "cess": "4%"},
                    ],
                    "rebate": {"max_income": 500000, "max_rebate": 12500},
                },
                {
                    "regime": "New",
                    "fy": "2024-25",
                    "ay": "2025-26",
                    "applicable_for": "Individuals (< 60 years)",
                    "slabs": [
                        {"from": 0, "to": 300000, "rate": 0, "cess": "4%"},
                        {"from": 300000, "to": 700000, "rate": 5, "cess": "4%"},
                        {"from": 700000, "to": 1000000, "rate": 10, "cess": "4%"},
                        {"from": 1000000, "to": 1200000, "rate": 15, "cess": "4%"},
                        {"from": 1200000, "to": 1500000, "rate": 20, "cess": "4%"},
                        {"from": 1500000, "to": None, "rate": 30, "cess": "4%"},
                    ],
                    "rebate": {"max_income": 700000, "max_rebate": 25000},
                },
            ],
            "capital_gains": [
                {
                    "type": "Long-Term Capital Gain (LTCG)",
                    "asset_class": "Equity (Listed shares & Equity Mutual Funds)",
                    "holding_period": "More than 12 months",
                    "tax_rate": 12.5,
                    "description": "LTCG on equity instruments held for more than 12 months is taxed at 12.5%. Exemption limit: ₹1.25 lakh per financial year. Grandfathering not applicable for purchases after 31 Jan 2018.",
                    "exemption_limit": 125000,
                    "indexation_available": False,
                },
                {
                    "type": "Short-Term Capital Gain (STCG)",
                    "asset_class": "Equity (Listed shares & Equity Mutual Funds)",
                    "holding_period": "12 months or less",
                    "tax_rate": 20,
                    "description": "STCG on equity instruments held for 12 months or less is taxed at 20%.",
                    "exemption_limit": None,
                    "indexation_available": False,
                },
                {
                    "type": "Long-Term Capital Gain (LTCG)",
                    "asset_class": "Debt (Debt Mutual Funds, Bonds, Fixed Deposits)",
                    "holding_period": "More than 24 months",
                    "tax_rate": 12.5,
                    "description": "LTCG on debt instruments held for more than 24 months is taxed at 12.5% without indexation benefit (as per Finance Act 2024).",
                    "exemption_limit": None,
                    "indexation_available": False,
                },
                {
                    "type": "Short-Term Capital Gain (STCG)",
                    "asset_class": "Debt (Debt Mutual Funds, Bonds, Fixed Deposits)",
                    "holding_period": "24 months or less",
                    "tax_rate": 0,
                    "description": "STCG on debt instruments is taxed at the individual's applicable slab rate. (rate=0 here denotes 'as per slab').",
                    "exemption_limit": None,
                    "indexation_available": False,
                },
                {
                    "type": "Long-Term Capital Gain (LTCG)",
                    "asset_class": "Real Estate (Immovable Property)",
                    "holding_period": "More than 24 months",
                    "tax_rate": 12.5,
                    "description": "LTCG on immovable property held for more than 24 months is taxed at 12.5% without indexation (post Budget 2024). Exemption available under Section 54/54EC.",
                    "exemption_limit": None,
                    "indexation_available": False,
                },
            ],
            "gst_rules": [
                {
                    "name": "Essential Commodities",
                    "category": "Essential",
                    "rate": 0,
                    "hsn_code": "Various",
                    "description": "Food grains, fresh fruits/vegetables, milk, eggs, etc. are exempt from GST.",
                    "exemptions": ["Unpacked food", "Fresh produce", "Milk and dairy"],
                    "itc_available": False,
                },
                {
                    "name": "Standard Rate Goods",
                    "category": "Standard",
                    "rate": 18,
                    "hsn_code": "Various",
                    "description": "Most manufactured goods and services fall under the 18% GST slab.",
                    "exemptions": [],
                    "itc_available": True,
                },
                {
                    "name": "Luxury / Sin Goods",
                    "category": "Luxury",
                    "rate": 28,
                    "hsn_code": "Various",
                    "description": "Luxury items like automobiles, aerated drinks, tobacco products, high-end appliances attract 28% GST with additional cess.",
                    "exemptions": [],
                    "itc_available": True,
                },
                {
                    "name": "Financial Services",
                    "category": "Services",
                    "rate": 18,
                    "hsn_code": "9971",
                    "description": "Banking, insurance, investment advisory, and other financial services attract 18% GST.",
                    "exemptions": ["Basic savings bank account services", "Life insurance (specific plans)", "Educational loans"],
                    "itc_available": True,
                },
            ],
            "deductions": [
                {
                    "name": "Section 80C - Life Insurance Premium",
                    "section": "80C",
                    "category": "insurance",
                    "max_limit": 150000,
                    "description": "Life insurance premium paid for self, spouse, or children is eligible for deduction under Section 80C (within the overall ₹1.5L limit).",
                    "eligible_investments": ["Life Insurance Premium", "ULIP", "Pension Plans"],
                    "conditions": ["Premium must not exceed 10% of sum assured for policies issued after 1 Apr 2012"],
                },
                {
                    "name": "Section 80E - Education Loan Interest",
                    "section": "80E",
                    "category": "education",
                    "max_limit": 0,  # No upper limit
                    "description": "Interest paid on education loan for higher studies is fully deductible for up to 8 years. No upper limit on the deduction amount.",
                    "eligible_investments": ["Education loan from approved financial institutions"],
                    "conditions": ["Loan must be for higher education", "Deduction available for 8 consecutive years starting from the year repayment begins"],
                },
            ],
        }
