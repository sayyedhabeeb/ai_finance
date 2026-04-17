"use client";

import React from "react";
import { Sidebar } from "@/components/Sidebar";
import { useUIStore } from "@/lib/store";

export function LayoutShell({ children }: { children: React.ReactNode }) {
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main
        className={`flex-1 overflow-y-auto transition-all duration-300 ease-in-out ${
          sidebarOpen ? "ml-64" : "ml-0 lg:ml-16"
        }`}
      >
        <div className="min-h-full">{children}</div>
      </main>
    </div>
  );
}
