"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, User, Copy, Check, ExternalLink, Brain, Shield, BarChart3, Zap } from "lucide-react";
import { clsx } from "clsx";
import type { ChatMessage as ChatMessageType } from "@/lib/types";
import { format } from "date-fns";

// ---------------------------------------------------------------------------
// Agent Icon Map
// ---------------------------------------------------------------------------
const agentIcons: Record<string, React.ReactNode> = {
  "Portfolio Agent": <BarChart3 className="h-3 w-3" />,
  "Risk Agent": <Shield className="h-3 w-3" />,
  "Market Data Agent": <Zap className="h-3 w-3" />,
  default: <Brain className="h-3 w-3" />,
};

const agentColors: Record<string, string> = {
  "Portfolio Agent": "bg-blue-500 bg-opacity-15 text-blue-400 border-blue-500 border-opacity-30",
  "Risk Agent": "bg-yellow-500 bg-opacity-15 text-yellow-400 border-yellow-500 border-opacity-30",
  "Market Data Agent": "bg-green-500 bg-opacity-15 text-green-400 border-green-500 border-opacity-30",
  default: "bg-purple-500 bg-opacity-15 text-purple-400 border-purple-500 border-opacity-30",
};

// ---------------------------------------------------------------------------
// Confidence Bar
// ---------------------------------------------------------------------------
function ConfidenceBar({ confidence }: { confidence: number }) {
  const percentage = Math.round(confidence * 100);
  const color =
    percentage >= 80
      ? "bg-accent-green"
      : percentage >= 60
      ? "bg-accent-yellow"
      : "bg-accent-red";

  return (
    <div className="flex items-center gap-2 text-xs text-gray-500">
      <span>Confidence:</span>
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-surface-300">
        <div
          className={`h-full rounded-full ${color} transition-all duration-500`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="font-medium">{percentage}%</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Source Card
// ---------------------------------------------------------------------------
function SourceCard({
  source,
}: {
  source: { title: string; url: string; snippet: string };
}) {
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex items-start gap-2.5 rounded-lg border border-surface-50 bg-surface-200 p-3 transition-all hover:border-surface-50 hover:bg-surface-50"
    >
      <ExternalLink className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-gray-500 group-hover:text-accent-blue transition-colors" />
      <div className="min-w-0">
        <p className="text-xs font-medium text-gray-300 group-hover:text-accent-blue transition-colors truncate">
          {source.title}
        </p>
        <p className="mt-0.5 text-[11px] text-gray-500 line-clamp-2">
          {source.snippet}
        </p>
      </div>
    </a>
  );
}

// ---------------------------------------------------------------------------
// Chat Message Component
// ---------------------------------------------------------------------------
interface ChatMessageProps {
  message: ChatMessageType;
  isStreaming?: boolean;
  onCopy?: () => void;
  isCopied?: boolean;
}

export function ChatMessage({
  message,
  isStreaming = false,
  onCopy,
  isCopied = false,
}: ChatMessageProps) {
  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";

  return (
    <div
      className={clsx(
        "animate-fade-in",
        isUser ? "flex justify-end" : "flex justify-start"
      )}
    >
      <div
        className={clsx(
          "flex gap-3 max-w-[85%]",
          isUser ? "flex-row-reverse" : "flex-row"
        )}
      >
        {/* Avatar */}
        <div
          className={clsx(
            "flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg",
            isUser
              ? "bg-accent-blue bg-opacity-10"
              : "bg-surface-100 border border-surface-50"
          )}
        >
          {isUser ? (
            <User className="h-4 w-4 text-accent-blue" />
          ) : (
            <Bot className="h-4 w-4 text-accent-blue" />
          )}
        </div>

        {/* Message Content */}
        <div className="flex flex-col gap-2 min-w-0">
          {/* Message Bubble */}
          <div
            className={clsx(
              "rounded-2xl px-4 py-3",
              isUser
                ? "bg-accent-blue text-white rounded-tr-sm"
                : "bg-surface-100 border border-surface-50 text-gray-200 rounded-tl-sm"
            )}
          >
            {isUser ? (
              <p className="text-sm leading-relaxed whitespace-pre-wrap">
                {message.content}
              </p>
            ) : (
              <div
                className={clsx(
                  "markdown-content text-sm",
                  isStreaming && "streaming-cursor"
                )}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content}
                </ReactMarkdown>
              </div>
            )}
          </div>

          {/* Meta Row (assistant only) */}
          {isAssistant && (
            <div className="flex flex-wrap items-center gap-2">
              {/* Timestamp */}
              <span className="text-[10px] text-gray-600">
                {message.timestamp
                  ? format(new Date(message.timestamp), "h:mm a")
                  : ""}
              </span>

              {/* Agent Badges */}
              {message.agents && message.agents.length > 0 && (
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] text-gray-600">Agents:</span>
                  {message.agents.map((agent) => (
                    <span
                      key={agent}
                      className={clsx(
                        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium",
                        agentColors[agent] || agentColors.default
                      )}
                    >
                      {agentIcons[agent] || agentIcons.default}
                      {agent}
                    </span>
                  ))}
                </div>
              )}

              {/* Copy Button */}
              {message.content && !isStreaming && (
                <button
                  onClick={onCopy}
                  className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] text-gray-500 hover:bg-surface-50 hover:text-gray-300 transition-colors"
                >
                  {isCopied ? (
                    <>
                      <Check className="h-3 w-3 text-accent-green" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      Copy
                    </>
                  )}
                </button>
              )}
            </div>
          )}

          {/* Confidence */}
          {isAssistant && message.confidence != null && !isStreaming && (
            <ConfidenceBar confidence={message.confidence} />
          )}

          {/* Sources */}
          {isAssistant &&
            message.sources &&
            message.sources.length > 0 &&
            !isStreaming && (
              <div className="space-y-2 mt-1">
                <p className="text-[10px] font-medium text-gray-500 uppercase tracking-wider">
                  Sources ({message.sources.length})
                </p>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {message.sources.map((source, i) => (
                    <SourceCard key={i} source={source} />
                  ))}
                </div>
              </div>
            )}
        </div>
      </div>
    </div>
  );
}
