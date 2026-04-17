"use client";

import React from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface AllocationData {
  name: string;
  value: number;
  color: string;
}

interface PerformanceDataPoint {
  date: string;
  portfolio: number;
  benchmark: number;
}

interface HoldingData {
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

interface PortfolioChartProps {
  type: "performance" | "allocation";
  allocations: AllocationData[];
  holdings: HoldingData[];
}

// ---------------------------------------------------------------------------
// Custom Tooltip
// ---------------------------------------------------------------------------
function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) {
  if (!active || !payload || !payload.length) return null;

  return (
    <div className="rounded-lg border border-surface-50 bg-surface-200 px-3 py-2 shadow-lg text-xs">
      <p className="text-gray-400 mb-1">{label}</p>
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2">
          <span
            className="h-2 w-2 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-gray-300">
            {entry.name}:{" "}
            <span className="font-medium text-white">
              {typeof entry.value === "number" && entry.name !== "Allocation"
                ? entry.value.toFixed(2)
                : `${entry.value.toFixed(1)}%`}
            </span>
          </span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pie Chart Tooltip
// ---------------------------------------------------------------------------
function PieTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; payload: { color: string } }>;
}) {
  if (!active || !payload || !payload.length) return null;

  const entry = payload[0];
  return (
    <div className="rounded-lg border border-surface-50 bg-surface-200 px-3 py-2 shadow-lg text-xs">
      <div className="flex items-center gap-2">
        <span
          className="h-2 w-2 rounded-full"
          style={{ backgroundColor: entry.payload.color }}
        />
        <span className="font-medium text-white">{entry.name}</span>
      </div>
      <p className="text-gray-400 mt-1">{entry.value.toFixed(1)}% of portfolio</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Generate Performance Data (mock data for demo)
// ---------------------------------------------------------------------------
function generatePerformanceData(): PerformanceDataPoint[] {
  const data: PerformanceDataPoint[] = [];
  let portfolioValue = 2700000;
  let benchmarkValue = 2700000;
  const now = new Date();

  for (let i = 89; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);

    // Random walk with slight upward drift
    portfolioValue *= 1 + (Math.random() - 0.48) * 0.015;
    benchmarkValue *= 1 + (Math.random() - 0.49) * 0.012;

    data.push({
      date: date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      }),
      portfolio: parseFloat((portfolioValue / 1000000).toFixed(3)),
      benchmark: parseFloat((benchmarkValue / 1000000).toFixed(3)),
    });
  }

  return data;
}

// ---------------------------------------------------------------------------
// Performance Chart
// ---------------------------------------------------------------------------
function PerformanceChart() {
  const data = generatePerformanceData();

  return (
    <div className="rounded-xl border border-surface-50 bg-surface-100 p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          Performance (90d)
        </h2>
        <div className="flex gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-accent-blue" />
            Portfolio
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-gray-500" />
            S&P 500
          </span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <defs>
            <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="benchmarkGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6b7280" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#6b7280" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#2a2f45"
            vertical={false}
          />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: "#64748b" }}
            axisLine={{ stroke: "#2a2f45" }}
            tickLine={false}
            interval={14}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#64748b" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => `$${v.toFixed(2)}M`}
            domain={["auto", "auto"]}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="portfolio"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#portfolioGradient)"
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0, fill: "#3b82f6" }}
          />
          <Area
            type="monotone"
            dataKey="benchmark"
            stroke="#6b7280"
            strokeWidth={1.5}
            strokeDasharray="4 4"
            fill="url(#benchmarkGradient)"
            dot={false}
            activeDot={{ r: 3, strokeWidth: 0, fill: "#6b7280" }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Allocation Chart
// ---------------------------------------------------------------------------
function AllocationChart({ allocations }: { allocations: AllocationData[] }) {
  const total = allocations.reduce((sum, a) => sum + a.value, 0);

  const data = allocations.map((a) => ({
    ...a,
    percentage: ((a.value / total) * 100).toFixed(1),
  }));

  return (
    <div className="rounded-xl border border-surface-50 bg-surface-100 p-5">
      <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">
        Sector Allocation
      </h2>
      <div className="flex flex-col lg:flex-row items-center gap-6">
        <div className="w-full lg:w-1/2">
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={3}
                dataKey="value"
                strokeWidth={0}
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<PieTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="w-full lg:w-1/2 space-y-3">
          {data.map((item) => (
            <div key={item.name} className="flex items-center gap-3">
              <span
                className="h-3 w-3 rounded-sm flex-shrink-0"
                style={{ backgroundColor: item.color }}
              />
              <span className="flex-1 text-sm text-gray-300">{item.name}</span>
              <div className="flex items-center gap-3">
                <div className="w-20 h-1.5 overflow-hidden rounded-full bg-surface-300">
                  <div
                    className="h-full rounded-full"
                    style={{
                      backgroundColor: item.color,
                      width: `${item.percentage}%`,
                    }}
                  />
                </div>
                <span className="text-xs font-medium text-gray-400 w-12 text-right">
                  {item.percentage}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main PortfolioChart Component (Router)
// ---------------------------------------------------------------------------
export function PortfolioChart({ type, allocations, holdings }: PortfolioChartProps) {
  if (type === "allocation") {
    return <AllocationChart allocations={allocations} />;
  }

  return <PerformanceChart />;
}

export default PortfolioChart;
