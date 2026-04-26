"use client";

import React, { useState, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Shield,
  Activity,
  ArrowRight,
  Brain,
  BarChart3,
  Zap,
} from "lucide-react";
import { apiClient } from "@/lib/api";
import { useChatStore } from "@/lib/store";

// ---------------------------------------------------------------------------
// Stat Card
// ---------------------------------------------------------------------------
interface StatCardProps {
  title: string;
  value: string;
  change?: string;
  changeType?: "positive" | "negative" | "neutral";
  icon: React.ReactNode;
  accentColor?: string;
}

function StatCard({
  title,
  value,
  change,
  changeType = "neutral",
  icon,
  accentColor = "border-accent-blue",
}: StatCardProps) {
  const changeColor =
    changeType === "positive"
      ? "text-accent-green"
      : changeType === "negative"
      ? "text-accent-red"
      : "text-gray-400";

  return (
    <div
      className={`rounded-xl border ${accentColor} border-opacity-20 bg-surface-100 p-5 transition-all duration-200 hover:border-opacity-40 hover:shadow-lg`}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-gray-400">{title}</p>
          <p className="mt-2 text-2xl font-bold tracking-tight">{value}</p>
          {change && (
            <p className={`mt-1 text-sm font-medium ${changeColor}`}>
              {changeType === "positive" && <TrendingUp className="mr-1 inline h-3 w-3" />}
              {changeType === "negative" && <TrendingDown className="mr-1 inline h-3 w-3" />}
              {change}
            </p>
          )}
        </div>
        <div className="rounded-lg bg-surface-200 p-2.5">{icon}</div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Mini Chart Sparkline (placeholder SVG)
// ---------------------------------------------------------------------------
function MiniSparkline({ trend }: { trend: "up" | "down" | "flat" }) {
  const color =
    trend === "up"
      ? "#22c55e"
      : trend === "down"
      ? "#ef4444"
      : "#eab308";

  const path =
    trend === "up"
      ? "M0,30 L8,25 L16,22 L24,18 L32,20 L40,12 L48,8"
      : trend === "down"
      ? "M0,8 L8,12 L16,15 L24,18 L32,16 L40,24 L48,30"
      : "M0,20 L8,18 L16,22 L24,19 L32,21 L40,20 L48,20";

  return (
    <svg viewBox="0 0 48 40" className="h-10 w-14" fill="none">
      <path
        d={path}
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Recent Activity Item
// ---------------------------------------------------------------------------
interface RecentActivityItem {
  id: string;
  type: "query" | "alert" | "trade" | "insight";
  title: string;
  description: string;
  time: string;
}

interface DashboardSummary {
  totalValue: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  riskScore: number;
  riskLevel: string;
}

function ActivityItem({ item }: { item: RecentActivityItem }) {
  const iconMap = {
    query: <Brain className="h-4 w-4 text-accent-blue" />,
    alert: <Shield className="h-4 w-4 text-accent-yellow" />,
    trade: <BarChart3 className="h-4 w-4 text-accent-purple" />,
    insight: <Zap className="h-4 w-4 text-accent-green" />,
  };

  return (
    <div className="flex items-start gap-3 rounded-lg p-3 transition-colors hover:bg-surface-50">
      <div className="mt-0.5 rounded-md bg-surface-200 p-1.5">
        {iconMap[item.type]}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-200 truncate">{item.title}</p>
        <p className="text-xs text-gray-500 mt-0.5">{item.description}</p>
      </div>
      <span className="shrink-0 text-xs text-gray-500">{item.time}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Dashboard Page
// ---------------------------------------------------------------------------
export default function DashboardPage() {
  const queryClient = useQueryClient();
  const addMessage = useChatStore((s) => s.addMessage);
  const [queryInput, setQueryInput] = useState("");

  // Fetch dashboard data
  const { data: portfolioData } = useQuery({
    queryKey: ["portfolio-summary"],
    queryFn: () => apiClient.getPortfolioSummary(),
  });

  const { data: recentActivity } = useQuery({
    queryKey: ["recent-activity"],
    queryFn: () => apiClient.getRecentActivity(),
  });

  // Quick query submission
  const handleQuickQuery = useCallback(() => {
    if (!queryInput.trim()) return;
    addMessage({
      id: crypto.randomUUID(),
      role: "user",
      content: queryInput,
      timestamp: new Date().toISOString(),
      agents: [],
      confidence: 0,
      metadata: {},
      sources: [],
    });
    setQueryInput("");
  }, [queryInput, addMessage]);

  const summary: DashboardSummary = portfolioData
    ? {
        totalValue: portfolioData.total_value,
        dailyPnL: portfolioData.daily_change ?? 0,
        dailyPnLPercent: 0,
        riskScore: 72,
        riskLevel: "Moderate",
      }
    : {
        totalValue: 2845921.45,
        dailyPnL: 12543.23,
        dailyPnLPercent: 0.44,
        riskScore: 72,
        riskLevel: "Moderate",
      };

  const activities: RecentActivityItem[] =
    (recentActivity as RecentActivityItem[] | undefined) ?? [
    {
      id: "1",
      type: "insight",
      title: "Portfolio rebalancing recommended",
      description: "Tech sector overweight by 12%. Consider reducing exposure.",
      time: "2m ago",
    },
    {
      id: "2",
      type: "alert",
      title: "Volatility spike detected",
      description: "VIX increased 15% in the last hour. Risk models updated.",
      time: "18m ago",
    },
    {
      id: "3",
      type: "query",
      title: "Analyzed AAPL earnings",
      description: "Beat estimates by $0.12. Revenue up 8% YoY.",
      time: "1h ago",
    },
    {
      id: "4",
      type: "trade",
      title: "Executed sector rotation",
      description: "Moved 5% from energy to healthcare sector.",
      time: "3h ago",
    },
    {
      id: "5",
      type: "insight",
      title: "Macro analysis complete",
      description: "Fed rate decision impact: moderate bullish bias for Q3.",
      time: "5h ago",
    },
  ];

  return (
    <div className="p-6 lg:p-8 space-y-8 animate-fade-in">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-400">
          Welcome back. Here&apos;s your financial overview.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          title="Portfolio Value"
          value={`$${(summary.totalValue).toLocaleString("en-US", { minimumFractionDigits: 2 })}`}
          change={`+$${summary.dailyPnL.toLocaleString("en-US", { minimumFractionDigits: 2 })} today`}
          changeType={summary.dailyPnL >= 0 ? "positive" : "negative"}
          icon={<DollarSign className="h-5 w-5 text-accent-blue" />}
          accentColor="border-accent-blue"
        />
        <StatCard
          title="Daily P&L"
          value={`$${summary.dailyPnL.toLocaleString("en-US", { minimumFractionDigits: 2 })}`}
          change={`${summary.dailyPnLPercent >= 0 ? "+" : ""}${summary.dailyPnLPercent.toFixed(2)}%`}
          changeType={summary.dailyPnLPercent >= 0 ? "positive" : "negative"}
          icon={
            summary.dailyPnL >= 0 ? (
              <TrendingUp className="h-5 w-5 text-accent-green" />
            ) : (
              <TrendingDown className="h-5 w-5 text-accent-red" />
            )
          }
          accentColor={
            summary.dailyPnLPercent >= 0
              ? "border-accent-green"
              : "border-accent-red"
          }
        />
        <StatCard
          title="Risk Score"
          value={`${summary.riskScore}/100`}
          change={summary.riskLevel}
          changeType="neutral"
          icon={<Shield className="h-5 w-5 text-accent-yellow" />}
          accentColor="border-accent-yellow"
        />
        <StatCard
          title="AI Queries Today"
          value="47"
          change="+12 vs yesterday"
          changeType="positive"
          icon={<Activity className="h-5 w-5 text-accent-purple" />}
          accentColor="border-accent-purple"
        />
      </div>

      {/* Quick Query + Performance Chart Row */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        {/* Quick Query */}
        <div className="rounded-xl border border-surface-50 bg-surface-100 p-5 xl:col-span-1">
          <h2 className="mb-4 text-sm font-semibold text-gray-300 uppercase tracking-wider">
            Quick Query
          </h2>
          <div className="space-y-3">
            <div className="relative">
              <input
                type="text"
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleQuickQuery()}
                placeholder="Ask about your portfolio, stocks, markets..."
                className="w-full rounded-lg border border-surface-50 bg-surface-300 px-4 py-3 pr-10 text-sm text-gray-200 placeholder-gray-500 transition-colors focus:border-accent-blue focus:outline-none focus:ring-1 focus:ring-accent-blue"
              />
              <button
                onClick={handleQuickQuery}
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-gray-400 transition-colors hover:bg-surface-200 hover:text-accent-blue"
              >
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {[
                "Analyze my risk exposure",
                "What's driving my portfolio today?",
                "Rebalance recommendations",
                "Market outlook this week",
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setQueryInput(suggestion)}
                  className="rounded-full border border-surface-50 bg-surface-200 px-3 py-1 text-xs text-gray-400 transition-colors hover:border-accent-blue hover:text-accent-blue"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Mini Performance Chart */}
        <div className="rounded-xl border border-surface-50 bg-surface-100 p-5 xl:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
              Performance (30d)
            </h2>
            <div className="flex gap-4 text-xs text-gray-500">
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-accent-blue" />
                Portfolio
              </span>
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-gray-500" />
                S&P 500
              </span>
            </div>
          </div>
          <div className="flex items-end justify-between h-32 gap-1">
            {Array.from({ length: 30 }).map((_, i) => {
              const portfolioH = 40 + Math.random() * 55;
              const benchmarkH = 35 + Math.random() * 50;
              return (
                <div key={i} className="flex-1 flex items-end gap-px">
                  <div
                    className="flex-1 rounded-t bg-accent-blue opacity-70"
                    style={{ height: `${portfolioH}%` }}
                  />
                  <div
                    className="flex-1 rounded-t bg-gray-600 opacity-40"
                    style={{ height: `${benchmarkH}%` }}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="rounded-xl border border-surface-50 bg-surface-100 p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
            Recent Activity
          </h2>
          <button className="text-xs text-accent-blue hover:underline">
            View all
          </button>
        </div>
        <div className="space-y-1">
          {activities.map((item) => (
            <ActivityItem key={item.id} item={item} />
          ))}
        </div>
      </div>
    </div>
  );
}
