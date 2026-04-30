import React from "react";
import type { Metadata } from "next";
import "./globals.css";
import { QueryProvider } from "@/lib/query-provider";
import { LayoutShell } from "./layout-shell";

export const metadata: Metadata = {
  title: "AI Financial Brain",
  description: "Intelligent financial analysis and portfolio management powered by AI",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const fontVars: React.CSSProperties = {
    ["--font-sans" as string]:
      "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
    ["--font-mono" as string]:
      "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace",
  };

  return (
    <html lang="en" className="dark">
      <body
        style={fontVars}
        className="font-sans antialiased bg-surface text-white"
      >
        <QueryProvider>
          <LayoutShell>{children}</LayoutShell>
        </QueryProvider>
      </body>
    </html>
  );
}
