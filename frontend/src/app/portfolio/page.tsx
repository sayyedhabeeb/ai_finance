"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  TrendingUp,
  TrendingDown,
  ArrowUpDown,
  Info,
} from "lucide-react";
import { apiClient } from "@/lib/api";
import { PortfolioChart } from "@/components/PortfolioChart";

// ---------------------------------------------------------------------------
// Holding Row
// ---------------------------------------------------------------------------
interface Holding {
  symbol: string;
  name: string;
  shares: number;
  price: number;
  change: number;
  changePercent: number;
  value: number;
  allocation: number;
  sector: string;
}

function HoldingsTable({ holdings }: { holdings: Holding[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-surface-50 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
            <th className="pb-3 pr-4">Symbol</th>
            <th className="pb-3 pr-4 hidden md:table-cell">Name</th>
            <th className="pb-3 pr-4 text-right">Shares</th>
            <th className="pb-3 pr-4 text-right">Price</th>
            <th className="pb-3 pr-4 text-right">Change</th>
            <th className="pb-3 pr-4 text-right">Value</th>
            <th className="pb-3 pr-4 text-right">Alloc %</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-surface-50">
          {holdings.map((h) => (
            <tr key={h.symbol} className="transition-colors hover:bg-surface-50">
              <td className="py-3 pr-4">
                <span className="font-semibold text-white">{h.symbol}</span>
              </td>
              <td className="py-3 pr-4 text-gray-400 hidden md:table-cell">
                {h.name}
              </td>
              <td className="py-3 pr-4 text-right text-gray-300">
                {h.shares.toLocaleString()}
              </td>
              <td className="py-3 pr-4 text-right text-gray-300">
                ${h.price.toFixed(2)}
              </td>
              <td
                className={`py-3 pr-4 text-right font-medium ${
                  h.change >= 0 ? "text-accent-green" : "text-accent-red"
                }`}
              >
                <span className="flex items-center justify-end gap-1">
                  {h.change >= 0 ? (
                    <TrendingUp className="h-3 w-3" />
                  ) : (
                    <TrendingDown className="h-3 w-3" />
                  )}
                  {h.change >= 0 ? "+" : ""}
                  {h.changePercent.toFixed(2)}%
                </span>
              </td>
              <td className="py-3 pr-4 text-right text-gray-200 font-medium">
                ${h.value.toLocaleString("en-US", { minimumFractionDigits: 2 })}
              </td>
              <td className="py-3 pr-4 text-right">
                <div className="flex items-center justify-end gap-2">
                  <div className="h-1.5 w-16 overflow-hidden rounded-full bg-surface-200">
                    <div
                      className="h-full rounded-full bg-accent-blue"
                      style={{ width: `${h.allocation}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-400 w-10 text-right">
                    {h.allocation.toFixed(1)}%
                  </span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Risk Metric Card
// ---------------------------------------------------------------------------
interface RiskMetric {
  label: string;
  value: string;
  description: string;
  status: "good" | "warning" | "danger";
}

function RiskMetricCard({ metric }: { metric: RiskMetric }) {
  const statusColors = {
    good: "border-accent-green bg-accent-green bg-opacity-5",
    warning: "border-accent-yellow bg-accent-yellow bg-opacity-5",
    danger: "border-accent-red bg-accent-red bg-opacity-5",
  };

  const statusDot = {
    good: "bg-accent-green",
    warning: "bg-accent-yellow",
    danger: "bg-accent-red",
  };

  return (
    <div
      className={`rounded-lg border-l-2 p-4 ${statusColors[metric.status]}`}
    >
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full ${statusDot[metric.status]}`} />
        <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">
          {metric.label}
        </p>
      </div>
      <p className="mt-1 text-lg font-bold text-white">{metric.value}</p>
      <p className="mt-0.5 text-xs text-gray-500">{metric.description}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Portfolio Page
// ---------------------------------------------------------------------------
export default function PortfolioPage() {
  const { data: portfolioData } = useQuery({
    queryKey: ["portfolio"],
    queryFn: () => apiClient.getPortfolio(),
  });

  const { data: riskData } = useQuery({
    queryKey: ["risk-metrics"],
    queryFn: () => apiClient.getRiskMetrics(),
  });

  const holdings: Holding[] = portfolioData?.holdings ?? [
    { symbol: "AAPL", name: "Apple Inc.", shares: 150, price: 189.84, change: 2.34, changePercent: 1.25, value: 28476.00, allocation: 18.2, sector: "Technology" },
    { symbol: "MSFT", name: "Microsoft Corp.", shares: 85, price: 420.55, change: 5.12, changePercent: 1.23, value: 35746.75, allocation: 22.8, sector: "Technology" },
    { symbol: "JPM", name: "JPMorgan Chase", shares: 120, price: 198.43, change: -1.87, changePercent: -0.93, value: 23811.60, allocation: 15.2, sector: "Financials" },
    { symbol: "JNJ", name: "Johnson & Johnson", shares: 200, price: 156.78, change: 0.92, changePercent: 0.59, value: 31356.00, allocation: 20.1, sector: "Healthcare" },
    { symbol: "XOM", name: "Exxon Mobil", shares: 180, price: 108.92, change: -2.15, changePercent: -1.94, value: 19605.60, allocation: 12.5, sector: "Energy" },
    { symbol: "BND", name: "Vanguard Total Bond", shares: 300, price: 72.45, change: 0.08, changePercent: 0.11, value: 21735.00, allocation: 13.9, sector: "Fixed Income" },
  ];

  const allocations = portfolioData?.allocations ?? [
    { name: "Technology", value: 41.0, color: "#3b82f6" },
    { name: "Healthcare", value: 20.1, color: "#22c55e" },
    { name: "Financials", value: 15.2, color: "#a855f7" },
    { name: "Energy", value: 12.5, color: "#eab308" },
    { name: "Fixed Income", value: 13.9, color: "#06b6d4" },
  ];

  const riskMetrics: RiskMetric[] = riskData ?? [
    { label: "Sharpe Ratio", value: "1.34", description: "Risk-adjusted return (good: > 1.0)", status: "good" },
    { label: "Value at Risk (95%)", value: "$42,350", description: "Maximum expected daily loss", status: "warning" },
    { label: "Portfolio Beta", value: "1.12", description: "Market sensitivity (1.0 = market)", status: "warning" },
    { label: "Max Drawdown", value: "-8.3%", description: "Largest peak-to-trough decline", status: "good" },
    { label: "Sortino Ratio", value: "1.67", description: "Downside risk-adjusted return", status: "good" },
    { label: "Diversification Score", value: "72/100", description: "Correlation-based diversification", status: "good" },
  ];

  const totalValue = holdings.reduce((sum, h) => sum + h.value, 0);
  const totalChange = holdings.reduce((sum, h) => sum + h.change * h.shares, 0);
  const totalChangePercent = (totalChange / (totalValue - totalChange)) * 100;

  return (
    <div className="p-6 lg:p-8 space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Portfolio</h1>
          <p className="mt-1 text-sm text-gray-400">
            Track holdings, allocations, and performance
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500">Last updated: just now</span>
          <button className="flex items-center gap-1.5 rounded-lg border border-surface-50 bg-surface-100 px-3 py-1.5 text-xs text-gray-300 hover:bg-surface-50 transition-colors">
            <ArrowUpDown className="h-3.5 w-3.5" />
            Rebalance
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="rounded-xl border border-surface-50 bg-surface-100 p-5">
          <p className="text-sm text-gray-400">Total Value</p>
          <p className="mt-1 text-2xl font-bold text-white">
            ${totalValue.toLocaleString("en-US", { minimumFractionDigits: 2 })}
          </p>
        </div>
        <div className="rounded-xl border border-surface-50 bg-surface-100 p-5">
          <p className="text-sm text-gray-400">Total Change</p>
          <div className="flex items-center gap-2 mt-1">
            <p
              className={`text-2xl font-bold ${
                totalChange >= 0 ? "text-accent-green" : "text-accent-red"
              }`}
            >
              {totalChange >= 0 ? "+" : ""}$
              {Math.abs(totalChange).toLocaleString("en-US", {
                minimumFractionDigits: 2,
              })}
            </p>
            <span
              className={`text-sm font-medium ${
                totalChangePercent >= 0
                  ? "text-accent-green"
                  : "text-accent-red"
              }`}
            >
              ({totalChangePercent >= 0 ? "+" : ""}
              {totalChangePercent.toFixed(2)}%)
            </span>
          </div>
        </div>
        <div className="rounded-xl border border-surface-50 bg-surface-100 p-5">
          <p className="text-sm text-gray-400">Number of Holdings</p>
          <p className="mt-1 text-2xl font-bold text-white">{holdings.length}</p>
          <p className="text-xs text-gray-500 mt-0.5">
            Across {new Set(holdings.map((h) => h.sector)).size} sectors
          </p>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <PortfolioChart
          type="performance"
          allocations={allocations}
          holdings={holdings}
        />
        <PortfolioChart
          type="allocation"
          allocations={allocations}
          holdings={holdings}
        />
      </div>

      {/* Holdings Table */}
      <div className="rounded-xl border border-surface-50 bg-surface-100 p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
            Holdings
          </h2>
          <button className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors">
            <Info className="h-3 w-3" />
            Export CSV
          </button>
        </div>
        <HoldingsTable holdings={holdings} />
      </div>

      {/* Risk Metrics */}
      <div className="rounded-xl border border-surface-50 bg-surface-100 p-5">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
          Risk Metrics
        </h2>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {riskMetrics.map((m) => (
            <RiskMetricCard key={m.label} metric={m} />
          ))}
        </div>
      </div>
    </div>
  );
}
