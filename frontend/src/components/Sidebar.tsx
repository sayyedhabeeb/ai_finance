"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  MessageSquare,
  PieChart,
  Target,
  Settings,
  ChevronLeft,
  ChevronRight,
  Brain,
  LogOut,
  Bell,
  User,
} from "lucide-react";
import { useUIStore, useUserStore } from "@/lib/store";
import { clsx } from "clsx";

// ---------------------------------------------------------------------------
// Navigation Items
// ---------------------------------------------------------------------------
interface NavItem {
  label: string;
  href: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  {
    label: "Dashboard",
    href: "/",
    icon: <LayoutDashboard className="h-5 w-5" />,
  },
  {
    label: "Chat",
    href: "/chat",
    icon: <MessageSquare className="h-5 w-5" />,
  },
  {
    label: "Portfolio",
    href: "/portfolio",
    icon: <PieChart className="h-5 w-5" />,
  },
  {
    label: "Goals",
    href: "/goals",
    icon: <Target className="h-5 w-5" />,
  },
];

const secondaryItems: NavItem[] = [
  {
    label: "Settings",
    href: "/settings",
    icon: <Settings className="h-5 w-5" />,
  },
];

// ---------------------------------------------------------------------------
// Sidebar Component
// ---------------------------------------------------------------------------
export function Sidebar() {
  const pathname = usePathname();
  const { sidebarOpen, toggleSidebar } = useUIStore();
  const { user, logout } = useUserStore();

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  };

  return (
    <aside
      className={clsx(
        "fixed left-0 top-0 z-40 flex h-full flex-col border-r border-surface-50 bg-surface-200 transition-all duration-300 ease-in-out",
        sidebarOpen ? "w-64" : "w-16"
      )}
    >
      {/* Logo Area */}
      <div className="flex h-16 items-center justify-between border-b border-surface-50 px-4">
        {sidebarOpen && (
          <Link href="/" className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-blue bg-opacity-10">
              <Brain className="h-5 w-5 text-accent-blue" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-white leading-tight">
                Financial Brain
              </h1>
              <p className="text-[10px] text-gray-500">AI-Powered</p>
            </div>
          </Link>
        )}
        {!sidebarOpen && (
          <Link href="/" className="flex items-center justify-center w-full">
            <Brain className="h-6 w-6 text-accent-blue" />
          </Link>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={clsx(
              "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all",
              isActive(item.href)
                ? "bg-accent-blue bg-opacity-10 text-accent-blue"
                : "text-gray-400 hover:bg-surface-50 hover:text-white"
            )}
            title={!sidebarOpen ? item.label : undefined}
          >
            <span
              className={clsx(
                "flex-shrink-0 transition-colors",
                isActive(item.href) ? "text-accent-blue" : "text-gray-500 group-hover:text-gray-300"
              )}
            >
              {item.icon}
            </span>
            {sidebarOpen && <span>{item.label}</span>}
          </Link>
        ))}
      </nav>

      {/* Secondary Navigation */}
      <div className="px-3 py-2 border-t border-surface-50">
        {secondaryItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={clsx(
              "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all",
              isActive(item.href)
                ? "bg-accent-blue bg-opacity-10 text-accent-blue"
                : "text-gray-400 hover:bg-surface-50 hover:text-white"
            )}
            title={!sidebarOpen ? item.label : undefined}
          >
            <span
              className={clsx(
                "flex-shrink-0 transition-colors",
                isActive(item.href) ? "text-accent-blue" : "text-gray-500 group-hover:text-gray-300"
              )}
            >
              {item.icon}
            </span>
            {sidebarOpen && <span>{item.label}</span>}
          </Link>
        ))}
      </div>

      {/* User Profile Area */}
      {sidebarOpen && user && (
        <div className="border-t border-surface-50 px-3 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-surface-300 text-sm font-semibold text-gray-300">
              {user.name
                ?.split(" ")
                .map((n) => n[0])
                .join("")
                .toUpperCase() || <User className="h-4 w-4" />}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-200 truncate">
                {user.name}
              </p>
              <p className="text-xs text-gray-500 truncate">{user.email}</p>
            </div>
            <button
              onClick={logout}
              className="rounded-lg p-1.5 text-gray-500 hover:bg-surface-50 hover:text-gray-300 transition-colors"
              title="Sign out"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* Collapse Toggle */}
      <button
        onClick={toggleSidebar}
        className="absolute -right-3 top-20 flex h-6 w-6 items-center justify-center rounded-full border border-surface-50 bg-surface-200 text-gray-400 hover:bg-surface-50 hover:text-white transition-colors shadow-sm"
      >
        {sidebarOpen ? (
          <ChevronLeft className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
      </button>
    </aside>
  );
}
