"use client";

import React from "react";
import { User, Bell, Shield, Palette, Database, Key } from "lucide-react";
import { useUserStore, useUIStore } from "@/lib/store";

export default function SettingsPage() {
  const { user } = useUserStore();
  const { sidebarOpen } = useUIStore();

  return (
    <div className="p-6 lg:p-8 space-y-8 animate-fade-in max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-white">Settings</h1>
        <p className="mt-1 text-sm text-gray-400">
          Manage your account and application preferences
        </p>
      </div>

      {/* Profile Section */}
      <div className="rounded-xl border border-surface-50 bg-surface-100 p-6">
        <div className="flex items-center gap-3 mb-6">
          <User className="h-4 w-4 text-accent-blue" />
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
            Profile
          </h2>
        </div>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Name</label>
            <input
              type="text"
              defaultValue={user?.name || ""}
              className="input"
              placeholder="Your name"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Email</label>
            <input
              type="email"
              defaultValue={user?.email || ""}
              className="input"
              placeholder="your@email.com"
            />
          </div>
        </div>
      </div>

      {/* API Keys Section */}
      <div className="rounded-xl border border-surface-50 bg-surface-100 p-6">
        <div className="flex items-center gap-3 mb-6">
          <Key className="h-4 w-4 text-accent-yellow" />
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
            API Keys
          </h2>
        </div>
        <p className="text-sm text-gray-400 mb-4">
          Configure your financial data provider API keys for real-time market data.
        </p>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Alpha Vantage Key</label>
            <input
              type="password"
              className="input"
              placeholder="Enter your Alpha Vantage API key"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Polygon.io Key</label>
            <input
              type="password"
              className="input"
              placeholder="Enter your Polygon API key"
            />
          </div>
        </div>
      </div>

      {/* Notifications */}
      <div className="rounded-xl border border-surface-50 bg-surface-100 p-6">
        <div className="flex items-center gap-3 mb-6">
          <Bell className="h-4 w-4 text-accent-purple" />
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
            Notifications
          </h2>
        </div>
        <div className="space-y-4">
          {[
            { label: "Portfolio alerts", description: "Get notified about significant portfolio changes" },
            { label: "Risk warnings", description: "Alert when risk metrics exceed thresholds" },
            { label: "Goal milestones", description: "Notify when you reach savings goals" },
            { label: "Market updates", description: "Daily market summary and relevant news" },
          ].map((item) => (
            <div key={item.label} className="flex items-center justify-between py-2">
              <div>
                <p className="text-sm font-medium text-gray-200">{item.label}</p>
                <p className="text-xs text-gray-500">{item.description}</p>
              </div>
              <button className="relative inline-flex h-6 w-11 items-center rounded-full bg-surface-300 transition-colors hover:bg-surface-50">
                <span className="inline-block h-4 w-4 transform rounded-full bg-gray-400 transition-transform translate-x-1" />
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className="flex justify-end gap-3">
        <button className="btn-secondary">Cancel</button>
        <button className="btn-primary">Save Changes</button>
      </div>
    </div>
  );
}
