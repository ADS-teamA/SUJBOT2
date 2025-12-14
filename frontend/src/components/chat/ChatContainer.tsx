/**
 * ChatContainer Component - Main chat area with messages and input
 *
 * Features:
 * - Welcome screen for new conversations
 * - Gradient background
 * - Animated input box transition (center → bottom on first message)
 */

import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { WelcomeScreen } from './WelcomeScreen';
import { ClarificationModal } from './ClarificationModal';
import { useCitationContext } from '../../contexts/CitationContext';
import { cn } from '../../design-system/utils/cn';
import type { Conversation, ClarificationData } from '../../types';
import type { SpendingLimitError } from '../../hooks/useChat';

interface ChatContainerProps {
  conversation: Conversation | undefined;
  isStreaming: boolean;
  onSendMessage: (message: string, addUserMessage?: boolean, selectedContext?: {
    text: string;
    documentId: string;
    documentName: string;
    pageStart: number;
    pageEnd: number;
  } | null) => void;
  onEditMessage: (messageId: string, newContent: string) => void;
  onRegenerateMessage: (messageId: string) => void;
  onCancelStreaming: () => void;
  clarificationData: ClarificationData | null;
  awaitingClarification: boolean;
  onSubmitClarification: (response: string) => void;
  onCancelClarification: () => void;
  spendingRefreshTrigger?: number;
  spendingLimitError?: SpendingLimitError | null;
  onClearSpendingLimitError?: () => void;
}

export function ChatContainer({
  conversation,
  isStreaming,
  onSendMessage,
  onEditMessage,
  onRegenerateMessage,
  onCancelStreaming,
  clarificationData,
  awaitingClarification,
  onSubmitClarification,
  onCancelClarification,
  spendingRefreshTrigger,
  spendingLimitError,
  onClearSpendingLimitError,
}: ChatContainerProps) {
  const { t, i18n } = useTranslation();
  const { selectedText, clearSelection } = useCitationContext();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [inputAnimated, setInputAnimated] = useState(false);
  const hasMessages = (conversation?.messages.length || 0) > 0;

  const containerRef = useRef<HTMLDivElement>(null);

  // Wrapper that includes selected context and clears selection after send
  const handleSendMessage = (message: string) => {
    const context = selectedText ? {
      text: selectedText.text,
      documentId: selectedText.documentId,
      documentName: selectedText.documentName,
      pageStart: selectedText.pageStart,
      pageEnd: selectedText.pageEnd,
    } : null;

    onSendMessage(message, true, context);

    // Auto-clear selection after sending (per user requirement)
    if (selectedText) {
      clearSelection();
    }
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation?.messages]);

  // Reset scroll when clearing messages (New Conversation)
  useEffect(() => {
    if (!hasMessages && containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [hasMessages]);

  // Trigger input animation on first message
  useEffect(() => {
    if (hasMessages && !inputAnimated) {
      setInputAnimated(true);
    }
  }, [hasMessages, inputAnimated]);

  return (
    <div className="flex-1 flex flex-col h-full relative">
      {/* Gradient background */}
      <div
        className={cn(
          'absolute inset-0 -z-10',
          'bg-white dark:bg-accent-950'
        )}
        style={{
          background: 'var(--gradient-mesh-light)',
        }}
      />
      <div
        className={cn(
          'absolute inset-0 -z-10',
          'dark:block hidden'
        )}
        style={{
          background: 'var(--gradient-mesh-dark)',
        }}
      />
      <div
        className={cn(
          'absolute inset-0 -z-10'
        )}
        style={{
          background: 'var(--gradient-light)',
        }}
      />
      <div
        className={cn(
          'absolute inset-0 -z-10',
          'dark:block hidden'
        )}
        style={{
          background: 'var(--gradient-dark)',
        }}
      />

      {/* Messages area */}
      <div
        ref={containerRef}
        className={cn(
          'flex-1',
          hasMessages ? 'overflow-y-auto' : 'overflow-hidden'
        )}>
        {!hasMessages ? (
          <WelcomeScreen onPromptClick={handleSendMessage}>
            {/* ChatInput as child - in natural document flow */}
            <div style={{ animation: 'fadeInScale 0.6s ease-out' }}>
              <ChatInput
                onSend={handleSendMessage}
                onCancel={onCancelStreaming}
                isStreaming={isStreaming}
                disabled={false}
                refreshSpendingTrigger={spendingRefreshTrigger}
                selectedText={selectedText}
                onClearSelection={clearSelection}
              />
            </div>
          </WelcomeScreen>
        ) : (
          <div
            className="max-w-5xl mx-auto py-4"
            style={{ animation: 'fadeIn 0.3s ease-out' }}
          >
            {conversation?.messages
              .filter((message) => {
                // Show user messages always
                if (message.role === 'user') return true;

                // Show assistant messages with:
                // 1. Non-empty content (after trimming), OR
                // 2. Tool calls (even if content is empty/whitespace), OR
                // 3. Active agent progress (during generation)
                const hasContent = message.content && message.content.trim().length > 0;
                const hasToolCalls = message.toolCalls && message.toolCalls.length > 0;
                const hasProgress = message.agentProgress && (
                  message.agentProgress.currentAgent !== null ||
                  message.agentProgress.activeTools?.length > 0
                );

                return hasContent || hasToolCalls || hasProgress;
              })
              .map((message, index, filteredMessages) => {
                // Calculate response duration for assistant messages
                let responseDurationMs: number | undefined;

                if (message.role === 'assistant' && index > 0) {
                  // Find the previous user message
                  const prevMessage = filteredMessages[index - 1];
                  if (prevMessage && prevMessage.role === 'user') {
                    const userTime = new Date(prevMessage.timestamp).getTime();
                    const assistantTime = new Date(message.timestamp).getTime();

                    // Validate timestamps are valid dates
                    if (isNaN(userTime)) {
                      console.error('Invalid user message timestamp:', prevMessage.timestamp);
                    } else if (isNaN(assistantTime)) {
                      console.error('Invalid assistant message timestamp:', message.timestamp);
                    } else {
                      const duration = assistantTime - userTime;

                      // Validate and warn about suspicious durations
                      if (duration < 0) {
                        // Negative duration (clock skew), don't show
                      } else if (duration > 24 * 60 * 60 * 1000) {
                        // > 24 hours, likely a regenerated message from old history
                        // Don't show duration as it's misleading
                      } else if (duration > 50) {
                        // Normal duration: > 50ms
                        // For very long durations (e.g. > 10 mins), it might be a regeneration
                        // of an old message, but we'll show it anyway as it might be useful context
                        // just without the console warning.
                        responseDurationMs = duration;
                      }
                      // Else: duration <= 50ms (likely cached/instant), don't show
                    }
                  }
                }

                return (
                  <div
                    key={message.id}
                    style={
                      index === 0 && inputAnimated
                        ? { animation: 'fadeInFromCenter 0.5s ease-out' }
                        : undefined
                    }
                  >
                    <ChatMessage
                      message={message}
                      animationDelay={index === 0 ? 0 : index * 100}
                      onEdit={onEditMessage}
                      onRegenerate={onRegenerateMessage}
                      disabled={isStreaming}
                      responseDurationMs={responseDurationMs}
                    />
                  </div>
                );
              })}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area - only shown when there are messages (welcome state has input inline) */}
      {hasMessages && (
        <div
          style={
            inputAnimated
              ? { animation: 'slideDown 0.4s ease-out' }
              : undefined
          }
        >
          <ChatInput
            onSend={handleSendMessage}
            onCancel={onCancelStreaming}
            isStreaming={isStreaming}
            disabled={false}
            refreshSpendingTrigger={spendingRefreshTrigger}
            selectedText={selectedText}
            onClearSelection={clearSelection}
          />
        </div>
      )}

      {/* HITL Clarification Modal */}
      <ClarificationModal
        isOpen={awaitingClarification}
        clarificationData={clarificationData}
        onSubmit={onSubmitClarification}
        onCancel={onCancelClarification}
        disabled={isStreaming}
      />

      {/* Spending Limit Error Modal */}
      {spendingLimitError && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={onClearSpendingLimitError}
          />
          {/* Modal */}
          <div className="relative bg-white dark:bg-accent-900 rounded-2xl shadow-2xl max-w-md w-full mx-4 p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 rounded-full bg-red-100 dark:bg-red-900/50 flex items-center justify-center">
                <span className="text-2xl">⚠️</span>
              </div>
              <div>
                <h2 className="text-xl font-semibold text-accent-900 dark:text-accent-100">
                  {t('chat.spendingLimitTitle')}
                </h2>
              </div>
            </div>
            <p className="text-accent-700 dark:text-accent-300 mb-4">
              {i18n.language === 'cs'
                ? spendingLimitError.message_cs
                : spendingLimitError.message_en}
            </p>
            <div className="bg-red-50 dark:bg-red-900/30 rounded-lg p-4 mb-6">
              <div className="flex justify-between items-center text-sm">
                <span className="text-accent-600 dark:text-accent-400">
                  {t('chat.spendingUsed')}:
                </span>
                <span className="font-semibold text-red-600 dark:text-red-400">
                  {spendingLimitError.total_spent_czk.toFixed(2)} / {spendingLimitError.spending_limit_czk.toFixed(2)} Kč
                </span>
              </div>
            </div>
            <button
              onClick={onClearSpendingLimitError}
              className={cn(
                'w-full py-3 px-4 rounded-xl font-medium',
                'bg-accent-900 dark:bg-accent-100',
                'text-white dark:text-accent-900',
                'hover:bg-accent-800 dark:hover:bg-accent-200',
                'transition-colors duration-200'
              )}
            >
              {t('common.ok')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
