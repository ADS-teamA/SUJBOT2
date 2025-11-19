/**
 * ChatContainer Component - Main chat area with messages and input
 *
 * Features:
 * - Welcome screen for new conversations
 * - Gradient background
 * - Animated input box transition (center → bottom on first message)
 */

import { useEffect, useRef, useState } from 'react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { WelcomeScreen } from './WelcomeScreen';
import { ClarificationModal } from './ClarificationModal';
import { cn } from '../../design-system/utils/cn';
import type { Conversation, ClarificationData } from '../../types';

interface ChatContainerProps {
  conversation: Conversation | undefined;
  isStreaming: boolean;
  onSendMessage: (message: string) => void;
  onEditMessage: (messageId: string, newContent: string) => void;
  onRegenerateMessage: (messageId: string) => void;
  onDeleteMessage: (messageId: string) => void;
  clarificationData: ClarificationData | null;
  awaitingClarification: boolean;
  onSubmitClarification: (response: string) => void;
  onCancelClarification: () => void;
}

export function ChatContainer({
  conversation,
  isStreaming,
  onSendMessage,
  onEditMessage,
  onRegenerateMessage,
  onDeleteMessage,
  clarificationData,
  awaitingClarification,
  onSubmitClarification,
  onCancelClarification,
}: ChatContainerProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [inputAnimated, setInputAnimated] = useState(false);
  const hasMessages = (conversation?.messages.length || 0) > 0;

  const containerRef = useRef<HTMLDivElement>(null);

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
          <WelcomeScreen onPromptClick={onSendMessage} />
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
                        console.warn('Negative duration detected (clock skew?):', {
                          userTime,
                          assistantTime,
                          duration
                        });
                        // Don't show negative durations (clock skew issue)
                      } else if (duration > 600000) {
                        // Backend took > 10 minutes - this indicates performance issues
                        console.error('⚠️ Backend response took > 10 minutes:', {
                          duration,
                          messageId: message.id,
                          durationMinutes: (duration / 60000).toFixed(1)
                        });
                        // Still show duration to user so they know backend is slow
                        responseDurationMs = duration;
                      } else if (duration > 50) {
                        // Normal duration: > 50ms, < 10 minutes
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
                      onDelete={onDeleteMessage}
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

      {/* Input area with animation */}
      <div
        className={cn(
          !hasMessages && 'absolute top-[40%] left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-3xl px-4'
        )}
        style={
          hasMessages && inputAnimated
            ? {
              animation: 'slideDown 0.4s ease-out',
            }
            : !hasMessages
              ? {
                animation: 'fadeInScale 0.6s ease-out',
              }
              : undefined
        }
      >
        <ChatInput onSend={onSendMessage} disabled={isStreaming} />
      </div>

      {/* HITL Clarification Modal */}
      <ClarificationModal
        isOpen={awaitingClarification}
        clarificationData={clarificationData}
        onSubmit={onSubmitClarification}
        onCancel={onCancelClarification}
        disabled={isStreaming}
      />
    </div>
  );
}
