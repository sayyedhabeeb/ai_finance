"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, User, Copy, Check } from "lucide-react";
import { clsx } from "clsx";
import type { ChatMessage as ChatMessageType } from "@/lib/types";
import { format } from "date-fns";

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
        </div>
      </div>
    </div>
  );
}
