"use client";

import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Plus,
  Target,
  TrendingUp,
  Calendar,
  DollarSign,
  Edit3,
  X,
  Check,
  Home,
  GraduationCap,
  Umbrella,
  Plane,
  Heart,
} from "lucide-react";
import { apiClient } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface Goal {
  id: string;
  name: string;
  description: string;
  icon: string;
  targetAmount: number;
  currentAmount: number;
  monthlyContribution: number;
  targetDate: string;
  category: string;
  status: "on_track" | "behind" | "completed";
  priority: "high" | "medium" | "low";
}

// ---------------------------------------------------------------------------
// Goal Icon Map
// ---------------------------------------------------------------------------
const iconMap: Record<string, React.ReactNode> = {
  home: <Home className="h-5 w-5" />,
  education: <GraduationCap className="h-5 w-5" />,
  retirement: <Umbrella className="h-5 w-5" />,
  travel: <Plane className="h-5 w-5" />,
  emergency: <Heart className="h-5 w-5" />,
  custom: <Target className="h-5 w-5" />,
};

const categoryColors: Record<string, { bg: string; text: string; border: string }> = {
  retirement: { bg: "bg-blue-500 bg-opacity-10", text: "text-blue-400", border: "border-blue-500 border-opacity-30" },
  home: { bg: "bg-purple-500 bg-opacity-10", text: "text-purple-400", border: "border-purple-500 border-opacity-30" },
  education: { bg: "bg-green-500 bg-opacity-10", text: "text-green-400", border: "border-green-500 border-opacity-30" },
  travel: { bg: "bg-yellow-500 bg-opacity-10", text: "text-yellow-400", border: "border-yellow-500 border-opacity-30" },
  emergency: { bg: "bg-red-500 bg-opacity-10", text: "text-red-400", border: "border-red-500 border-opacity-30" },
  custom: { bg: "bg-gray-500 bg-opacity-10", text: "text-gray-400", border: "border-gray-500 border-opacity-30" },
};

// ---------------------------------------------------------------------------
// Goal Card
// ---------------------------------------------------------------------------
function GoalCard({ goal }: { goal: Goal }) {
  const progress = (goal.currentAmount / goal.targetAmount) * 100;
  const colors = categoryColors[goal.category] || categoryColors.custom;
  const statusColors = {
    on_track: "text-accent-green",
    behind: "text-accent-yellow",
    completed: "text-accent-blue",
  };

  const daysRemaining = Math.max(
    0,
    Math.ceil(
      (new Date(goal.targetDate).getTime() - Date.now()) / (1000 * 60 * 60 * 24)
    )
  );

  return (
    <div
      className={`rounded-xl border ${colors.border} ${colors.bg} p-5 transition-all duration-200 hover:shadow-lg`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={`rounded-lg ${colors.bg} p-2 ${colors.text}`}>
            {iconMap[goal.icon] || iconMap.custom}
          </div>
          <div>
            <h3 className="font-semibold text-white">{goal.name}</h3>
            <p className="text-xs text-gray-400 mt-0.5">{goal.description}</p>
          </div>
        </div>
        <button className="rounded-lg p-1.5 text-gray-500 hover:bg-surface-50 hover:text-gray-300 transition-colors">
          <Edit3 className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Progress */}
      <div className="mt-4">
        <div className="flex items-end justify-between mb-1.5">
          <span className="text-lg font-bold text-white">
            ${goal.currentAmount.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
          </span>
          <span className="text-xs text-gray-400">
            of ${goal.targetAmount.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-surface-300">
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              goal.status === "completed"
                ? "bg-accent-blue"
                : goal.status === "on_track"
                ? "bg-accent-green"
                : "bg-accent-yellow"
            }`}
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>
        <p className="mt-1 text-xs text-gray-500">
          {progress.toFixed(1)}% complete
        </p>
      </div>

      {/* Details */}
      <div className="mt-4 grid grid-cols-3 gap-3 text-center">
        <div>
          <DollarSign className="mx-auto h-3.5 w-3.5 text-gray-500" />
          <p className="mt-1 text-xs font-medium text-gray-300">
            ${goal.monthlyContribution.toLocaleString()}/mo
          </p>
          <p className="text-[10px] text-gray-500">Monthly</p>
        </div>
        <div>
          <Calendar className="mx-auto h-3.5 w-3.5 text-gray-500" />
          <p className="mt-1 text-xs font-medium text-gray-300">
            {daysRemaining > 0 ? `${daysRemaining}d` : "Done"}
          </p>
          <p className="text-[10px] text-gray-500">Remaining</p>
        </div>
        <div>
          <TrendingUp className="mx-auto h-3.5 w-3.5 text-gray-500" />
          <p className={`mt-1 text-xs font-medium capitalize ${statusColors[goal.status]}`}>
            {goal.status.replace("_", " ")}
          </p>
          <p className="text-[10px] text-gray-500">Status</p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Add Goal Form
// ---------------------------------------------------------------------------
function AddGoalForm({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const [form, setForm] = useState({
    name: "",
    description: "",
    icon: "custom",
    targetAmount: "",
    currentAmount: "0",
    monthlyContribution: "",
    targetDate: "",
    category: "custom",
    priority: "medium" as "high" | "medium" | "low",
  });

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // In production, call apiClient.createGoal(form)
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60 backdrop-blur-sm">
      <div className="mx-4 w-full max-w-lg rounded-2xl border border-surface-50 bg-surface-100 p-6 animate-slide-up">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-white">New Financial Goal</h2>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-gray-400 hover:bg-surface-50 hover:text-gray-200 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Goal Name
            </label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              className="w-full rounded-lg border border-surface-50 bg-surface-300 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-accent-blue focus:outline-none focus:ring-1 focus:ring-accent-blue"
              placeholder="e.g., Dream Home Down Payment"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Description
            </label>
            <input
              type="text"
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })}
              className="w-full rounded-lg border border-surface-50 bg-surface-300 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-accent-blue focus:outline-none focus:ring-1 focus:ring-accent-blue"
              placeholder="Brief description of your goal"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Target Amount ($)
              </label>
              <input
                type="number"
                value={form.targetAmount}
                onChange={(e) => setForm({ ...form, targetAmount: e.target.value })}
                className="w-full rounded-lg border border-surface-50 bg-surface-300 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-accent-blue focus:outline-none focus:ring-1 focus:ring-accent-blue"
                placeholder="100000"
                required
                min="0"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Monthly Contribution ($)
              </label>
              <input
                type="number"
                value={form.monthlyContribution}
                onChange={(e) =>
                  setForm({ ...form, monthlyContribution: e.target.value })
                }
                className="w-full rounded-lg border border-surface-50 bg-surface-300 px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:border-accent-blue focus:outline-none focus:ring-1 focus:ring-accent-blue"
                placeholder="2000"
                required
                min="0"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Target Date
              </label>
              <input
                type="date"
                value={form.targetDate}
                onChange={(e) => setForm({ ...form, targetDate: e.target.value })}
                className="w-full rounded-lg border border-surface-50 bg-surface-300 px-3 py-2 text-sm text-gray-200 focus:border-accent-blue focus:outline-none focus:ring-1 focus:ring-accent-blue"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Category
              </label>
              <select
                value={form.category}
                onChange={(e) => {
                  setForm({ ...form, category: e.target.value, icon: e.target.value });
                }}
                className="w-full rounded-lg border border-surface-50 bg-surface-300 px-3 py-2 text-sm text-gray-200 focus:border-accent-blue focus:outline-none focus:ring-1 focus:ring-accent-blue"
              >
                <option value="retirement">Retirement</option>
                <option value="home">Home</option>
                <option value="education">Education</option>
                <option value="travel">Travel</option>
                <option value="emergency">Emergency Fund</option>
                <option value="custom">Custom</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Priority
            </label>
            <div className="flex gap-3">
              {(["high", "medium", "low"] as const).map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setForm({ ...form, priority: p })}
                  className={`rounded-lg border px-4 py-1.5 text-xs font-medium capitalize transition-colors ${
                    form.priority === p
                      ? "border-accent-blue bg-accent-blue bg-opacity-10 text-accent-blue"
                      : "border-surface-50 text-gray-400 hover:border-gray-600"
                  }`}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          <div className="flex gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 rounded-lg border border-surface-50 py-2.5 text-sm font-medium text-gray-300 hover:bg-surface-50 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 rounded-lg bg-accent-blue py-2.5 text-sm font-medium text-white hover:bg-blue-600 transition-colors"
            >
              Create Goal
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Goals Page
// ---------------------------------------------------------------------------
export default function GoalsPage() {
  const [showAddForm, setShowAddForm] = useState(false);

  const { data: goalsData } = useQuery({
    queryKey: ["goals"],
    queryFn: () => apiClient.getGoals(),
  });

  const goals: Goal[] = goalsData ?? [
    {
      id: "1",
      name: "Retirement Fund",
      description: "Financial independence by age 60",
      icon: "retirement",
      targetAmount: 2000000,
      currentAmount: 845000,
      monthlyContribution: 3000,
      targetDate: "2045-06-01",
      category: "retirement",
      status: "on_track",
      priority: "high",
    },
    {
      id: "2",
      name: "Dream Home",
      description: "Down payment for a 4-bedroom house",
      icon: "home",
      targetAmount: 150000,
      currentAmount: 67500,
      monthlyContribution: 2500,
      targetDate: "2026-12-01",
      category: "home",
      status: "on_track",
      priority: "high",
    },
    {
      id: "3",
      name: "Kids Education",
      description: "College fund for two children",
      icon: "education",
      targetAmount: 300000,
      currentAmount: 85000,
      monthlyContribution: 1500,
      targetDate: "2035-09-01",
      category: "education",
      status: "behind",
      priority: "high",
    },
    {
      id: "4",
      name: "Emergency Fund",
      description: "6 months of expenses saved",
      icon: "emergency",
      targetAmount: 50000,
      currentAmount: 42500,
      monthlyContribution: 500,
      targetDate: "2025-03-01",
      category: "emergency",
      status: "on_track",
      priority: "medium",
    },
    {
      id: "5",
      name: "World Trip",
      description: "Extended family vacation",
      icon: "travel",
      targetAmount: 25000,
      currentAmount: 18000,
      monthlyContribution: 800,
      targetDate: "2025-08-01",
      category: "travel",
      status: "on_track",
      priority: "low",
    },
  ];

  const totalTarget = goals.reduce((s, g) => s + g.targetAmount, 0);
  const totalCurrent = goals.reduce((s, g) => s + g.currentAmount, 0);
  const overallProgress = (totalCurrent / totalTarget) * 100;

  return (
    <div className="p-6 lg:p-8 space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Financial Goals</h1>
          <p className="mt-1 text-sm text-gray-400">
            Track and manage your financial targets
          </p>
        </div>
        <button
          onClick={() => setShowAddForm(true)}
          className="flex items-center gap-2 rounded-lg bg-accent-blue px-4 py-2 text-sm font-medium text-white hover:bg-blue-600 transition-colors"
        >
          <Plus className="h-4 w-4" />
          Add Goal
        </button>
      </div>

      {/* Overall Summary */}
      <div className="rounded-xl border border-surface-50 bg-surface-100 p-6">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
            Overall Progress
          </h2>
          <span className="text-2xl font-bold text-white">
            {overallProgress.toFixed(1)}%
          </span>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-surface-300">
          <div
            className="h-full rounded-full bg-gradient-to-r from-accent-blue to-accent-green transition-all duration-1000"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
        <div className="flex items-center justify-between mt-3 text-sm">
          <span className="text-gray-400">
            ${totalCurrent.toLocaleString()} saved
          </span>
          <span className="text-gray-500">
            Goal: ${totalTarget.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Goal Cards Grid */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {goals.map((goal) => (
          <GoalCard key={goal.id} goal={goal} />
        ))}
      </div>

      {/* AI Recommendations */}
      <div className="rounded-xl border border-accent-blue border-opacity-20 bg-accent-blue bg-opacity-5 p-5">
        <div className="flex items-center gap-2 mb-3">
          <Target className="h-4 w-4 text-accent-blue" />
          <h2 className="text-sm font-semibold text-accent-blue uppercase tracking-wider">
            AI Recommendations
          </h2>
        </div>
        <ul className="space-y-2 text-sm text-gray-300">
          <li className="flex items-start gap-2">
            <Check className="mt-0.5 h-4 w-4 shrink-0 text-accent-green" />
            <span>
              Your <strong>Kids Education</strong> fund is behind schedule.
              Consider increasing monthly contributions by $200 to stay on track.
            </span>
          </li>
          <li className="flex items-start gap-2">
            <Check className="mt-0.5 h-4 w-4 shrink-0 text-accent-green" />
            <span>
              <strong>Emergency Fund</strong> is nearly complete. Once funded,
              redirect the $500/month to your <strong>Retirement Fund</strong> for
              compound growth benefits.
            </span>
          </li>
          <li className="flex items-start gap-2">
            <Check className="mt-0.5 h-4 w-4 shrink-0 text-accent-green" />
            <span>
              With current market conditions, your <strong>Retirement Fund</strong>{" "}
              allocation could benefit from a 5% shift toward index funds for
              lower fees and better long-term returns.
            </span>
          </li>
        </ul>
      </div>

      <AddGoalForm isOpen={showAddForm} onClose={() => setShowAddForm(false)} />
    </div>
  );
}
