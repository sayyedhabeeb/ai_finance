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
import { apiClient } from "@/lib/api";

const SUGGESTIONS = [
  "Analyse the risk of investing in HDFC Bank right now",
  "How is Nifty 50 performing this week?",
  "Recommend risk hedging strategies for my portfolio",
  "What is the outlook for IT sector stocks?",
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

  // STABLE session ID — persists across all messages in this tab
  const sessionId = useRef<string>(crypto.randomUUID());

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const sendQuery = useCallback(
    async (queryText: string) => {
      setIsStreaming(true);
      const assistantMessageId = crypto.randomUUID();
      addMessage({
        id: assistantMessageId,
        role: "assistant" as const,
        content: "...",
        timestamp: new Date().toISOString(),
        agents: [],
        confidence: 0,
        metadata: {},
        sources: [],
      });

      try {
        const data = await apiClient.query(
          queryText,
          "anonymous",
          sessionId.current  // stable — AI remembers context
        );

        const content =
          data.response ??
          data.answer ??
          data.message ??
          data.final_response ??
          "No response received.";

        const confidence =
          typeof data.confidence === "number" ? data.confidence : 0;

        const agentList = Array.isArray(data.agents_used)
          ? data.agents_used
          : data.agent_type
            ? [data.agent_type]
            : ["financial_analyst"];

        const sourceList = Array.isArray(data.sources) ? data.sources : [];

        useChatStore.setState((state) => ({
          messages: state.messages.map((m) =>
            m.id === assistantMessageId
              ? {
                  ...m,
                  content,
                  agents: agentList,
                  confidence,
                  metadata: data.metadata ?? {},
                  sources: sourceList.map((s: string) => ({
                    title: s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, " "),
                    url: "#",
                    snippet: `Information retrieved via ${s}.`,
                  })),
                }
              : m
          ),
        }));
      } catch (error: any) {
        useChatStore.setState((state) => ({
          messages: state.messages.map((m) =>
            m.id === assistantMessageId
              ? {
                  ...m,
                  content: `Error: ${error.message}. Please ensure the backend is running on port 8000.`,
                }
              : m
          ),
        }));
      } finally {
        setIsStreaming(false);
      }
    },
    [addMessage, setIsStreaming]
  );

  const handleSend = useCallback(() => {
    const trimmedInput = input.trim();
    if (!trimmedInput || isStreaming) return;
    addMessage({
      id: crypto.randomUUID(),
      role: "user" as const,
      content: trimmedInput,
      timestamp: new Date().toISOString(),
      agents: [],
      confidence: 0,
      metadata: {},
      sources: [],
    });
    setInput("");
    sendQuery(trimmedInput);
  }, [input, isStreaming, addMessage, sendQuery]);

  const handleCopy = useCallback((text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  }, []);

  // Reset session on New Chat
  const handleNewChat = useCallback(() => {
    clearMessages();
    sessionId.current = crypto.randomUUID();
  }, [clearMessages]);

  return (
    <div className="flex h-screen flex-col">
      <div className="flex items-center justify-between border-b border-surface-50 bg-surface-200 px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-accent-blue bg-opacity-10 p-2">
            <Bot className="h-5 w-5 text-accent-blue" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white">AI Financial Brain</h1>
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
          onClick={handleNewChat}
          className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs text-gray-400 transition-colors hover:bg-surface-100 hover:text-gray-200"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          New chat
        </button>
      </div>

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
              I can analyse your portfolio, provide market insights, assess risk,
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
              <button className="rounded-lg p-1.5 text-gray-500 transition-colors hover:bg-surface-50 hover:text-gray-300" title="Attach file">
                <Paperclip className="h-4 w-4" />
              </button>
              <button className="rounded-lg p-1.5 text-gray-500 transition-colors hover:bg-surface-50 hover:text-gray-300" title="Voice input">
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
            AI Financial Brain may make mistakes. Verify important financial decisions with a qualified advisor.
          </p>
        </div>
      </div>
    </div>
  );
}
