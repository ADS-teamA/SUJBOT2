/**
 * ChatMessage Component - Displays a single message (user or assistant)
 */

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { User, Bot, Clock, DollarSign, Edit2, RotateCw, Check, X } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';
import { useSlideIn } from '../../design-system/animations/hooks/useSlideIn';
import type { Message } from '../../types';
import { ToolCallDisplay } from './ToolCallDisplay';
import { AgentProgress } from './AgentProgress';

interface ChatMessageProps {
  message: Message;
  animationDelay?: number;
  onEdit: (messageId: string, newContent: string) => void;
  onRegenerate: (messageId: string) => void;
  disabled?: boolean;
  responseDurationMs?: number; // Duration in milliseconds for assistant responses
}

export function ChatMessage({
  message,
  animationDelay = 0,
  onEdit,
  onRegenerate,
  disabled = false,
  responseDurationMs,
}: ChatMessageProps) {
  const isUser = message.role === 'user';
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(message.content);

  const { style: slideStyle } = useSlideIn({
    direction: 'up',
    delay: animationDelay,
    duration: 'normal',
  });

  const handleEdit = () => {
    setIsEditing(true);
    setEditedContent(message.content);
  };

  const handleSaveEdit = () => {
    if (editedContent.trim() && editedContent !== message.content) {
      onEdit(message.id, editedContent.trim());
    }
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditedContent(message.content);
  };

  const handleRegenerate = () => {
    onRegenerate(message.id);
  };

  return (
    <div
      style={slideStyle}
      className={cn(
        'flex gap-4 p-4',
        'transition-shadow duration-300',
        'hover:shadow-md',
        isUser
          ? 'bg-accent-50 dark:bg-accent-900/50'
          : 'bg-white dark:bg-accent-950'
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          'flex-shrink-0 w-8 h-8 rounded-full',
          'flex items-center justify-center',
          'transition-transform duration-200',
          'hover:scale-110',
          isUser
            ? 'bg-accent-700 dark:bg-accent-300 text-white dark:text-accent-900'
            : 'bg-accent-800 dark:bg-accent-200 text-accent-100 dark:text-accent-900'
        )}
      >
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        {/* Role label and actions */}
        <div className="flex items-center justify-between mb-2">
          <div className={cn(
            'text-sm font-medium',
            'text-accent-700 dark:text-accent-300'
          )}>
            {isUser ? 'You' : 'Assistant'}
          </div>

          {/* Action buttons */}
          {!disabled && !isEditing && (
            <div className="flex gap-1">
              {isUser && (
                <button
                  onClick={handleEdit}
                  className={cn(
                    'p-1.5 rounded',
                    'text-accent-500 hover:text-accent-700',
                    'dark:text-accent-400 dark:hover:text-accent-200',
                    'hover:bg-accent-100 dark:hover:bg-accent-800',
                    'transition-colors'
                  )}
                  title="Edit message"
                >
                  <Edit2 size={14} />
                </button>
              )}
              {!isUser && (
                <button
                  onClick={handleRegenerate}
                  className={cn(
                    'p-1.5 rounded',
                    'text-accent-500 hover:text-accent-700',
                    'dark:text-accent-400 dark:hover:text-accent-200',
                    'hover:bg-accent-100 dark:hover:bg-accent-800',
                    'transition-colors'
                  )}
                  title="Regenerate response"
                >
                  <RotateCw size={14} />
                </button>
              )}
            </div>
          )}
        </div>

        {/* Message content or editor */}
        {isEditing ? (
          <div className="space-y-2">
            <textarea
              value={editedContent}
              onChange={(e) => setEditedContent(e.target.value)}
              className={cn(
                'w-full p-3 rounded-lg border',
                'border-accent-300 dark:border-accent-700',
                'bg-white dark:bg-accent-900',
                'text-accent-900 dark:text-accent-100',
                'focus:outline-none focus:ring-2',
                'focus:ring-accent-500 dark:focus:ring-accent-400',
                'resize-none'
              )}
              rows={4}
              autoFocus
            />
            <div className="flex gap-2">
              <button
                onClick={handleSaveEdit}
                disabled={!editedContent.trim()}
                className={cn(
                  'px-3 py-1.5 rounded flex items-center gap-1.5',
                  'bg-accent-700 hover:bg-accent-800',
                  'dark:bg-accent-600 dark:hover:bg-accent-700',
                  'text-white text-sm font-medium',
                  'disabled:opacity-50 disabled:cursor-not-allowed',
                  'transition-colors'
                )}
              >
                <Check size={14} />
                Save & Send
              </button>
              <button
                onClick={handleCancelEdit}
                className={cn(
                  'px-3 py-1.5 rounded flex items-center gap-1.5',
                  'bg-accent-200 hover:bg-accent-300',
                  'dark:bg-accent-800 dark:hover:bg-accent-700',
                  'text-accent-900 dark:text-accent-100 text-sm font-medium',
                  'transition-colors'
                )}
              >
                <X size={14} />
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* Agent progress for assistant messages - collapsible */}
            {!isUser && message.agentProgress && (
              <details className={cn(
                'mb-3 group',
                'border border-accent-200 dark:border-accent-700',
                'rounded-lg overflow-hidden',
                'transition-colors'
              )}>
                <summary className={cn(
                  'px-3 py-2 cursor-pointer',
                  'bg-accent-50 dark:bg-accent-900/50',
                  'hover:bg-accent-100 dark:hover:bg-accent-800',
                  'text-accent-600 dark:text-accent-400',
                  'text-xs font-medium',
                  'flex items-center gap-2',
                  'select-none',
                  'transition-colors',
                  '[&::-webkit-details-marker]:hidden' // Hide default marker
                )}>
                  <span className={cn(
                    'text-accent-500 dark:text-accent-500',
                    'transition-transform duration-200',
                    'group-open:rotate-90'
                  )}>▸</span>
                  <span className="group-open:hidden">Show agent progress</span>
                  <span className="hidden group-open:inline">Hide agent progress</span>
                </summary>

                <div className={cn(
                  'bg-white dark:bg-accent-950',
                  'border-t border-accent-200 dark:border-accent-700'
                )}>
                  <AgentProgress progress={message.agentProgress} />
                </div>
              </details>
            )}

            {/* Message content */}
            <div className="prose dark:prose-invert prose-sm max-w-none">
              {(() => {
              // Display strategy:
              // - If text contains [Using ...] markers → inline rendering (tools shown inline with text)
              // - If NO markers but toolCalls exist → fallback rendering (tools shown at top)
              // This prevents duplicate display (tools at top + inline)

              const hasMarkers = /\[Using\s+[^\]]+\.\.\.\]/.test(message.content);
              const hasToolCalls = message.toolCalls && message.toolCalls.length > 0;

              // Fallback: Show tools at top if no markers (backward compatibility)
              if (!hasMarkers && hasToolCalls) {
                return (
                  <>
                    {/* Tool calls without inline markers - show at top */}
                    <div className="space-y-2 mb-3">
                      {message.toolCalls!.map((toolCall) => (
                        <ToolCallDisplay key={toolCall.id} toolCall={toolCall} />
                      ))}
                    </div>
                    {/* Regular markdown content */}
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                      components={{
                        code({ node, inline, className, children, ...props }: any) {
                          return inline ? (
                            <code
                              className={cn(
                                'px-1 py-0.5 rounded text-sm',
                                'bg-accent-100 dark:bg-accent-800'
                              )}
                              {...props}
                            >
                              {children}
                            </code>
                          ) : (
                            <code className={className} {...props}>
                              {children}
                            </code>
                          );
                        },
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </>
                );
              }

              // Primary path: Inline rendering with markers
              // Inline tool call display with content parsing
              //
              // Edge case handling:
              // 1. Empty responses can occur when LLM uses only tools without explanation
              // 2. [Using ...] markers are UI placeholders during streaming, removed after
              // 3. Valid states:
              //    - Content only (no tools): Normal text response
              //    - Content + tools: Response with tool usage (most common)
              //    - Tools only: Tool-only response (valid, no error)
              //    - Neither: Error state (LLM bug or streaming failure)

              // Check if content has substance (after removing [Using ...] markers)
              // Remove markers and collapse all whitespace (including Unicode nbsp, zero-width)
              const contentWithoutMarkers = message.content
                .replace(/\[Using [^\]]+\.\.\.\]\n*/g, '')
                .trim()
                .replace(/\s+/g, '');  // Collapse all whitespace

              const hasContent = contentWithoutMarkers.length > 0;

              // Error: Neither content nor tools (shouldn't happen, but defensive)
              if (!isUser && !hasContent && !hasToolCalls) {
                return (
                  <div className={cn(
                    'px-3 py-2 rounded',
                    'bg-accent-100 dark:bg-accent-800',
                    'text-accent-700 dark:text-accent-300',
                    'text-sm italic'
                  )}>
                    ⚠️ Model returned empty response. This may indicate an API error. Try regenerating.
                  </div>
                );
              }

              // Inline tool display parsing strategy
              //
              // Format: Assistant response contains "[Using tool_name...]" markers where tools were called
              // Goal: Split text around markers and insert actual ToolCallDisplay components inline
              //
              // Regex: /\[Using ([^\]]+)\.\.\.\]\n*/g
              //   - [Using ...] - Literal marker format
              //   - ([^\]]+) - Capture group: tool name (any chars except ])
              //   - \.\.\.\] - Literal "...]"
              //   - \n* - Optional trailing newlines (normalize whitespace)
              //
              // Matching strategy: Match markers by tool NAME (not ID) because:
              //   1. Markers are inserted during streaming before we have tool IDs
              //   2. Multiple calls to same tool must be matched in order (usedToolCallIndices tracking)
              //   3. If tool call missing for marker, we skip silently (defensive - shouldn't happen)
              const toolMarkerRegex = /\[Using ([^\]]+)\.\.\.\]\n*/g;
              const parts: React.JSX.Element[] = [];
              let lastIndex = 0;
              let match;

              // Track matched tool calls to handle duplicate tool names (e.g., search called twice)
              const usedToolCallIndices = new Set<number>();

              while ((match = toolMarkerRegex.exec(message.content)) !== null) {
                // Add text before the marker
                if (match.index > lastIndex) {
                  const textBefore = message.content.substring(lastIndex, match.index);
                  parts.push(
                    <ReactMarkdown
                      key={`text-${lastIndex}`}
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                      components={{
                        code({ node, inline, className, children, ...props }: any) {
                          return inline ? (
                            <code
                              className={cn(
                                'px-1 py-0.5 rounded text-sm',
                                'bg-accent-100 dark:bg-accent-800'
                              )}
                              {...props}
                            >
                              {children}
                            </code>
                          ) : (
                            <code className={className} {...props}>
                              {children}
                            </code>
                          );
                        },
                      }}
                    >
                      {textBefore}
                    </ReactMarkdown>
                  );
                }

                // Add tool call display - find first unused tool call with matching name
                const toolName = match[1];
                const toolCallIndex = message.toolCalls?.findIndex(
                  (tc, idx) => tc.name === toolName && !usedToolCallIndices.has(idx)
                );

                if (toolCallIndex !== undefined && toolCallIndex >= 0 && message.toolCalls) {
                  const toolCall = message.toolCalls[toolCallIndex];
                  usedToolCallIndices.add(toolCallIndex);

                  parts.push(
                    <div key={`tool-${toolCall.id}`} className="my-3">
                      <ToolCallDisplay toolCall={toolCall} />
                    </div>
                  );
                }

                lastIndex = match.index + match[0].length;
              }

              // Add remaining text after last marker
              if (lastIndex < message.content.length) {
                const textAfter = message.content.substring(lastIndex);
                parts.push(
                  <ReactMarkdown
                    key={`text-${lastIndex}`}
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeHighlight]}
                    components={{
                      code({ node, inline, className, children, ...props }: any) {
                        return inline ? (
                          <code
                            className={cn(
                              'px-1 py-0.5 rounded text-sm',
                              'bg-accent-100 dark:bg-accent-800'
                            )}
                            {...props}
                          >
                            {children}
                          </code>
                        ) : (
                          <code className={className} {...props}>
                            {children}
                          </code>
                        );
                      },
                    }}
                  >
                    {textAfter}
                  </ReactMarkdown>
                );
              }

              // If no markers found, render as before
              if (parts.length === 0) {
                return (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeHighlight]}
                    components={{
                      code({ node, inline, className, children, ...props }: any) {
                        return inline ? (
                          <code
                            className={cn(
                              'px-1 py-0.5 rounded text-sm',
                              'bg-accent-100 dark:bg-accent-800'
                            )}
                            {...props}
                          >
                            {children}
                          </code>
                        ) : (
                          <code className={className} {...props}>
                            {children}
                          </code>
                        );
                      },
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                );
              }

              return <>{parts}</>;
            })()}
            </div>
          </>
        )}

        {/* Collapsible metadata section (cost, duration, tool usage) */}
        {!isUser && (message.cost?.totalCost !== undefined || responseDurationMs !== undefined || (message.toolCalls && message.toolCalls.length > 0)) && (
          <details className={cn(
            'mt-3 group',
            'border border-accent-200 dark:border-accent-700',
            'rounded-lg overflow-hidden',
            'transition-colors'
          )}>
            <summary className={cn(
              'px-3 py-2 cursor-pointer',
              'bg-accent-50 dark:bg-accent-900/50',
              'hover:bg-accent-100 dark:hover:bg-accent-800',
              'text-accent-600 dark:text-accent-400',
              'text-xs font-medium',
              'flex items-center gap-2',
              'select-none',
              'transition-colors',
              '[&::-webkit-details-marker]:hidden' // Hide default marker
            )}>
              <span className={cn(
                'text-accent-500 dark:text-accent-500',
                'transition-transform duration-200',
                'group-open:rotate-90'
              )}>▸</span>
              <span className="group-open:hidden">Show execution details</span>
              <span className="hidden group-open:inline">Hide execution details</span>
            </summary>

            <div className={cn(
              'px-3 py-2 space-y-2',
              'bg-white dark:bg-accent-950',
              'border-t border-accent-200 dark:border-accent-700'
            )}>
              {/* Cost information */}
              {message.cost && message.cost.totalCost !== undefined && (
                <div className={cn(
                  'flex items-center gap-4 text-xs',
                  'text-accent-500 dark:text-accent-400'
                )}>
                  <span className="flex items-center gap-1">
                    <DollarSign size={12} />
                    ${message.cost.totalCost.toFixed(4)}
                  </span>
                  {message.cost.inputTokens !== undefined && message.cost.outputTokens !== undefined && (
                    <span>
                      {message.cost.inputTokens.toLocaleString()} in /{' '}
                      {message.cost.outputTokens.toLocaleString()} out
                    </span>
                  )}
                  {message.cost.cachedTokens !== undefined && message.cost.cachedTokens > 0 && (
                    <span className={cn(
                      'text-accent-600 dark:text-accent-400'
                    )}>
                      {message.cost.cachedTokens.toLocaleString()} cached
                    </span>
                  )}
                </div>
              )}

              {/* Response Duration */}
              {responseDurationMs !== undefined && (
                <div className={cn(
                  'flex items-center gap-1 text-xs',
                  'text-accent-400 dark:text-accent-500'
                )}>
                  <Clock size={12} />
                  Response time: {(responseDurationMs / 1000).toFixed(2)}s
                </div>
              )}

              {/* Tool usage count */}
              {message.toolCalls && message.toolCalls.length > 0 && (
                <div className={cn(
                  'text-xs',
                  'text-accent-600 dark:text-accent-400'
                )}>
                  Tools used: {message.toolCalls.length}
                </div>
              )}
            </div>
          </details>
        )}

        {/* Timestamp (always visible for user messages) */}
        {isUser && (
          <div className={cn(
            'mt-2 flex items-center gap-1 text-xs',
            'text-accent-400 dark:text-accent-500'
          )}>
            <Clock size={12} />
            {(() => {
              const date = message.timestamp ? new Date(message.timestamp) : new Date();
              return !isNaN(date.getTime()) ? date.toLocaleTimeString() : 'Just now';
            })()}
          </div>
        )}
      </div>
    </div>
  );
}
