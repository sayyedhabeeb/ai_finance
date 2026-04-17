"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  Bot,
  Sparkles,
  RotateCcw,
  Paperclip,
  Mic,
} from "lucide-react";
import { useChatStore } from "@/lib/store";
import { ChatMessage } from "@/components/ChatMessage";

// ---------------------------------------------------------------------------
// Follow-up Suggestion Chips
// ---------------------------------------------------------------------------
const SUGGESTIONS = [
  "What is my portfolio's Sharpe ratio?",
  "How is tech sector performing this quarter?",
  "Recommend risk hedging strategies",
  "Compare my allocation to the optimal one",
  "What are the biggest risks in my portfolio?",
];

export default function ChatPage() {
  const {
    messages,
    addMessage,
    isStreaming,
    setIsStreaming,
    clearMessages,
  } = useChatStore();
  const [input, setInput] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  // Simulated streaming response
  const simulateStreamResponse = useCallback(
    async () => {
      setIsStreaming(true);

      // Simulate agent processing
      const agents = ["Portfolio Agent", "Risk Agent", "Market Data Agent"];
      const usedAgents = agents.slice(
        0,
        1 + Math.floor(Math.random() * agents.length)
      );

      const assistantMessage = {
        id: crypto.randomUUID(),
        role: "assistant" as const,
        content: "",
        timestamp: new Date().toISOString(),
        agents: usedAgents,
        sources: [
          {
            title: "S&P 500 Daily Report",
            url: "https://example.com/report/sp500",
            snippet: "S&P 500 gained 1.2% on strong earnings reports from major tech companies.",
          },
          {
            title: "Portfolio Risk Analysis",
            url: "https://example.com/report/risk",
            snippet: "Current VaR at 95% confidence: $42,350. Portfolio beta: 1.12.",
          },
        ],
        confidence: 0.87,
      };

      addMessage(assistantMessage);

      // Simulate word-by-word streaming
      const fullResponse = `Based on my analysis of your current portfolio and market conditions, here are my findings:\n\n## Portfolio Summary\n\nYour portfolio is currently **well-diversified** across 8 sectors with a total value of **$2,845,921.45**. The risk-adjusted return (Sharpe ratio) stands at **1.34**, which is above the benchmark.\n\n## Key Observations\n\n1. **Technology sector** is your largest allocation at 28.5% — this is 5% above your target allocation\n2. **Healthcare** is underweight by 3% and showing strong momentum\n3. Your portfolio's **beta of 1.12** means it moves roughly in line with the market\n\n## Recommendations\n\nI suggest rebalancing by:\n- Reducing tech exposure by 3-5%\n- Increasing healthcare allocation by 2-3%\n- Adding 1-2% to fixed income for downside protection\n\nThe current market conditions favor a **moderately bullish** stance with caution around upcoming Fed decisions.\n\n*Analysis based on data from ${usedAgents.join(", ")}.*`;

      const words = fullResponse.split(" ");
      for (let i = 0; i < words.length; i++) {
        await new Promise((r) => setTimeout(r, 20 + Math.random() * 30));
        useChatStore.setState((state) => ({
          messages: state.messages.map((m) =>
            m.id === assistantMessage.id
              ? { ...m, content: words.slice(0, i + 1).join(" ") }
              : m
          ),
        }));
      }

      setIsStreaming(false);
    },
    [addMessage, setIsStreaming]
  );

  const handleSend = useCallback(() => {
    if (!input.trim() || isStreaming) return;

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user" as const,
      content: input.trim(),
      timestamp: new Date().toISOString(),
    };

    addMessage(userMessage);
    setInput("");
    simulateStreamResponse();
  }, [input, isStreaming, addMessage, simulateStreamResponse]);

  const handleCopy = useCallback((text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  }, []);

  return (
    <div className="flex h-screen flex-col">
      {/* Chat Header */}
      <div className="flex items-center justify-between border-b border-surface-50 bg-surface-200 px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-accent-blue bg-opacity-10 p-2">
            <Bot className="h-5 w-5 text-accent-blue" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white">
              AI Financial Brain
            </h1>
            <p className="text-xs text-gray-400">
              {isStreaming ? (
                <span className="flex items-center gap-1">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-green" />
                  Thinking...
                </span>
              ) : (
                "Ready to analyze"
              )}
            </p>
          </div>
        </div>
        <button
          onClick={clearMessages}
          className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs text-gray-400 transition-colors hover:bg-surface-100 hover:text-gray-200"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          New chat
        </button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <div className="mb-6 rounded-2xl bg-surface-100 p-6">
              <Sparkles className="h-10 w-10 text-accent-blue" />
            </div>
            <h2 className="text-xl font-semibold text-white mb-2">
              Ask me anything about your finances
            </h2>
            <p className="mb-8 max-w-md text-sm text-gray-400">
              I can analyze your portfolio, provide market insights, assess risk,
              and help with financial planning decisions.
            </p>
            <div className="flex flex-wrap justify-center gap-2 max-w-lg">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setInput(suggestion)}
                  className="rounded-full border border-surface-50 bg-surface-100 px-4 py-2 text-sm text-gray-300 transition-all hover:border-accent-blue hover:bg-surface-50 hover:text-accent-blue"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-6">
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                isStreaming={
                  isStreaming &&
                  message.role === "assistant" &&
                  message === messages[messages.length - 1]
                }
                onCopy={() => handleCopy(message.content, message.id)}
                isCopied={copiedId === message.id}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-surface-50 bg-surface-200 px-6 py-4">
        <div className="mx-auto max-w-3xl">
          <div className="relative rounded-xl border border-surface-50 bg-surface-300 transition-colors focus-within:border-accent-blue">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Ask about your portfolio, stocks, risk analysis..."
              rows={1}
              className="w-full resize-none rounded-xl bg-transparent px-4 py-3 pr-24 text-sm text-gray-200 placeholder-gray-500 focus:outline-none"
            />
            <div className="absolute right-2 bottom-2 flex items-center gap-1">
              <button
                className="rounded-lg p-1.5 text-gray-500 transition-colors hover:bg-surface-50 hover:text-gray-300"
                title="Attach file"
              >
                <Paperclip className="h-4 w-4" />
              </button>
              <button
                className="rounded-lg p-1.5 text-gray-500 transition-colors hover:bg-surface-50 hover:text-gray-300"
                title="Voice input"
              >
                <Mic className="h-4 w-4" />
              </button>
              <button
                onClick={handleSend}
                disabled={!input.trim() || isStreaming}
                className="ml-1 rounded-lg bg-accent-blue p-2 text-white transition-all hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </div>
          <p className="mt-2 text-center text-xs text-gray-600">
            AI Financial Brain may make mistakes. Verify important financial
            decisions with a qualified advisor.
          </p>
        </div>
      </div>
    </div>
  );
}
