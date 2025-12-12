/**
 * ChatMessage Component - Bubble-based message display (Claude/OpenAI style)
 *
 * Features:
 * - Asymmetric bubble layout (user right, assistant left)
 * - Inline editing with smooth transitions
 * - Collapsible tool calls and metadata
 * - Markdown rendering with syntax highlighting
 */

import React, { useState, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import { Clock, DollarSign, Edit2, RotateCw, Check, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import type { Message } from '../../types';
import { ToolCallDisplay } from './ToolCallDisplay';
import { ProgressPhaseDisplay } from './ProgressPhaseDisplay';
import { CitationLink } from '../citation/CitationLink';
import { preprocessCitations } from '../../utils/citations';

/**
 * Custom markdown components including citation support.
 * The cite component renders CitationLink for <cite data-chunk-id="..."> elements.
 */
const createMarkdownComponents = () => ({
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
  // Citation handler: renders CitationLink for <cite> elements
  cite({ node, ...props }: any) {
    const chunkId = props['data-chunk-id'];
    if (chunkId) {
      return <CitationLink chunkId={chunkId} />;
    }
    // Fallback for cite without data-chunk-id
    return <cite {...props} />;
  },
});

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
  animationDelay: _animationDelay = 0,
  onEdit,
  onRegenerate,
  disabled = false,
  responseDurationMs,
}: ChatMessageProps) {
  const { t } = useTranslation();
  const isUser = message.role === 'user';
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(message.content);

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

  // Preprocess content to convert \cite{chunk_id} to HTML <cite> tags
  // Only for assistant messages (user messages don't have citations)
  const processedContent = useMemo(() => {
    if (isUser) return message.content;
    return preprocessCitations(message.content);
  }, [message.content, isUser]);

  // Memoize markdown components to avoid recreation on each render
  const markdownComponents = useMemo(() => createMarkdownComponents(), []);

  return (
    <div
      className={cn(
        'flex w-full px-4 py-6',
        isUser ? 'justify-end' : 'justify-start'
      )}
      style={{
        animation: `fadeIn 0.3s ease-out`,
      }}
    >
      <div
        className={cn(
          'max-w-[85%] md:max-w-[75%] lg:max-w-[65%]',
          'space-y-2'
        )}
      >
        {/* Role label (small, subtle) */}
        <div
          className={cn(
            'text-xs font-medium tracking-wide uppercase px-1',
            'text-accent-500 dark:text-accent-500',
            isUser ? 'text-right' : 'text-left'
          )}
        >
          {isUser ? t('chat.you') : t('chat.assistant')}
        </div>

        {/* Message bubble */}
        <div
          className={cn(
            'rounded-2xl px-5 py-4',
            'shadow-sm',
            'transition-all duration-300',
            isUser
              ? cn(
                'bg-accent-900 dark:bg-accent-100',
                'text-accent-50 dark:text-accent-900',
                'rounded-tr-sm' // Distinctive corner
              )
              : cn(
                'bg-white dark:bg-accent-900',
                'text-accent-900 dark:text-accent-100',
                'border border-accent-200 dark:border-accent-800',
                'rounded-tl-sm' // Distinctive corner
              )
          )}
        >
          {/* Message content or editor */}
          {isEditing ? (
            <div className="space-y-3">
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
                autoComplete="off"
                spellCheck={false}
                data-lpignore="true"
              />
              <div className="flex gap-2">
                <button
                  onClick={handleSaveEdit}
                  disabled={!editedContent.trim()}
                  className={cn(
                    'px-3 py-1.5 rounded-lg flex items-center gap-1.5',
                    'bg-accent-700 hover:bg-accent-800',
                    'dark:bg-accent-600 dark:hover:bg-accent-700',
                    'text-white text-sm font-medium',
                    'disabled:opacity-50 disabled:cursor-not-allowed',
                    'transition-colors'
                  )}
                >
                  <Check size={14} />
                  {t('chat.saveAndSend')}
                </button>
                <button
                  onClick={handleCancelEdit}
                  className={cn(
                    'px-3 py-1.5 rounded-lg flex items-center gap-1.5',
                    'bg-accent-200 hover:bg-accent-300',
                    'dark:bg-accent-800 dark:hover:bg-accent-700',
                    'text-accent-900 dark:text-accent-100 text-sm font-medium',
                    'transition-colors'
                  )}
                >
                  <X size={14} />
                  {t('chat.cancel')}
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* Agent progress for assistant messages - show while streaming or agent active */}
              {!isUser && (message.agentProgress?.currentAgent || message.agentProgress?.isStreaming) && (
                <div className="mb-4">
                  <ProgressPhaseDisplay progress={message.agentProgress} />
                </div>
              )}

              {/* Message content */}
              <div
                className={cn(
                  'prose prose-sm max-w-none',
                  isUser
                    ? 'prose-invert dark:prose'
                    : 'prose dark:prose-invert',
                  'prose-headings:font-display prose-headings:font-medium',
                  'prose-p:leading-relaxed'
                )}
              >
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
                        {/* Regular markdown content with citation support */}
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          rehypePlugins={[rehypeRaw, rehypeHighlight]}
                          components={markdownComponents}
                        >
                          {processedContent}
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
                  // Only show if NOT currently processing (no active agent)
                  if (!isUser && !hasContent && !hasToolCalls && !message.agentProgress?.currentAgent) {
                    return (
                      <div className={cn(
                        'px-3 py-2 rounded',
                        'bg-accent-100 dark:bg-accent-800',
                        'text-accent-700 dark:text-accent-300',
                        'text-sm italic'
                      )}>
                        ⚠️ {t('chat.emptyResponse')}
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

                  while ((match = toolMarkerRegex.exec(processedContent)) !== null) {
                    // Add text before the marker
                    if (match.index > lastIndex) {
                      const textBefore = processedContent.substring(lastIndex, match.index);
                      parts.push(
                        <ReactMarkdown
                          key={`text-${lastIndex}`}
                          remarkPlugins={[remarkGfm]}
                          rehypePlugins={[rehypeRaw, rehypeHighlight]}
                          components={markdownComponents}
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
                  if (lastIndex < processedContent.length) {
                    const textAfter = processedContent.substring(lastIndex);
                    parts.push(
                      <ReactMarkdown
                        key={`text-${lastIndex}`}
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeRaw, rehypeHighlight]}
                        components={markdownComponents}
                      >
                        {textAfter}
                      </ReactMarkdown>
                    );
                  }

                  // If no markers found, render with citation support
                  if (parts.length === 0) {
                    return (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeRaw, rehypeHighlight]}
                        components={markdownComponents}
                      >
                        {processedContent}
                      </ReactMarkdown>
                    );
                  }

                  return <>{parts}</>;
                })()}
              </div>
            </>
          )}
        </div>

        {/* Action buttons and timestamp (outside bubble) */}
        <div
          className={cn(
            'flex items-center gap-2 px-1',
            isUser ? 'justify-end' : 'justify-start'
          )}
        >
          {/* Action buttons */}
          {!disabled && !isEditing && (
            <div className="flex gap-1">
              {isUser && (
                <button
                  onClick={handleEdit}
                  className={cn(
                    'p-1.5 rounded-lg',
                    'text-accent-500 hover:text-accent-700',
                    'dark:text-accent-400 dark:hover:text-accent-200',
                    'hover:bg-accent-100 dark:hover:bg-accent-800',
                    'transition-colors'
                  )}
                  title={t('chat.editMessage')}
                >
                  <Edit2 size={14} />
                </button>
              )}
              {!isUser && (
                <button
                  onClick={handleRegenerate}
                  className={cn(
                    'p-1.5 rounded-lg',
                    'text-accent-500 hover:text-accent-700',
                    'dark:text-accent-400 dark:hover:text-accent-200',
                    'hover:bg-accent-100 dark:hover:bg-accent-800',
                    'transition-colors'
                  )}
                  title={t('chat.regenerate')}
                >
                  <RotateCw size={14} />
                </button>
              )}
            </div>
          )}

          {/* Timestamp with inline details toggle */}
          <div
            className={cn(
              'flex items-center gap-1 text-xs',
              'text-accent-400 dark:text-accent-600'
            )}
          >
            <Clock size={12} />
            {(() => {
              const date = message.timestamp ? new Date(message.timestamp) : new Date();
              return !isNaN(date.getTime()) ? date.toLocaleTimeString() : t('common.justNow');
            })()}

            {/* Minimalist details chevron (only for assistant messages with metadata) */}
            {!isUser && (message.cost?.totalCost !== undefined || responseDurationMs !== undefined || (message.toolCalls && message.toolCalls.length > 0)) && (
              <>
                <span className="text-accent-300 dark:text-accent-700">•</span>
                <details className="group relative inline-block">
                  <summary className={cn(
                    'cursor-pointer select-none',
                    'text-accent-400 dark:text-accent-600',
                    'hover:text-accent-600 dark:hover:text-accent-400',
                    'transition-colors',
                    'list-none [&::-webkit-details-marker]:hidden',
                    'inline-flex items-center gap-0.5'
                  )}>
                    <span className={cn(
                      'transition-transform duration-200',
                      'group-open:rotate-90',
                      'text-xs leading-none'
                    )}>▸</span>
                    <span className="text-xs leading-none">{t('chat.details')}</span>
                  </summary>

                  {/* Dropdown panel below */}
                  <div className={cn(
                    'absolute left-0 mt-1 z-10',
                    'min-w-[300px]',
                    'border border-accent-200 dark:border-accent-700',
                    'rounded-lg overflow-hidden',
                    'shadow-lg',
                    'bg-white dark:bg-accent-950',
                    'px-3 py-2 space-y-2',
                    'text-xs'
                  )}>
                    {/* Cost information with per-agent breakdown */}
                    {message.cost && message.cost.totalCost !== undefined && (
                      <div className="space-y-2">
                        {/* Total cost */}
                        <div className={cn(
                          'flex items-center gap-4 text-xs',
                          'text-accent-500 dark:text-accent-400'
                        )}>
                          <span className="flex items-center gap-1 font-medium">
                            <DollarSign size={12} />
                            ${message.cost.totalCost.toFixed(4)}
                          </span>
                          {message.cost.inputTokens !== undefined && message.cost.outputTokens !== undefined && (
                            <span>
                              {message.cost.inputTokens.toLocaleString()} {t('chat.tokensIn')} /{' '}
                              {message.cost.outputTokens.toLocaleString()} {t('chat.tokensOut')}
                            </span>
                          )}
                          {message.cost.cachedTokens !== undefined && message.cost.cachedTokens > 0 && (
                            <span className={cn(
                              'text-accent-600 dark:text-accent-400'
                            )}>
                              {message.cost.cachedTokens.toLocaleString()} {t('chat.cached')}
                            </span>
                          )}
                        </div>

                        {/* Per-agent breakdown (if available) */}
                        {message.cost.agentBreakdown && message.cost.agentBreakdown.length > 0 && (
                          <details className={cn(
                            'text-xs',
                            'text-accent-500 dark:text-accent-400'
                          )}>
                            <summary className={cn(
                              'cursor-pointer select-none',
                              'hover:text-accent-700 dark:hover:text-accent-300',
                              'transition-colors'
                            )}>
                              <span className="inline-block w-3">▸</span>
                              {t('chat.perAgentBreakdown')} ({message.cost.agentBreakdown.length} {t('chat.agents')})
                            </summary>
                            <div className="mt-2 ml-3 space-y-1">
                              {message.cost.agentBreakdown.map((agent, idx) => {
                                // Defensive rendering with fallbacks
                                const agentName = agent?.agent || 'Unknown';
                                const cost = typeof agent?.cost === 'number' ? agent.cost : 0;
                                const inputTokens = typeof agent?.input_tokens === 'number' ? agent.input_tokens : 0;
                                const outputTokens = typeof agent?.output_tokens === 'number' ? agent.output_tokens : 0;
                                const cacheTokens = typeof agent?.cache_read_tokens === 'number' ? agent.cache_read_tokens : 0;

                                // Skip rendering if agent data is completely invalid
                                if (!agent || (!agentName && cost === 0)) {
                                  console.warn('Skipping invalid agent cost data:', agent);
                                  return null;
                                }

                                const responseTime = typeof agent?.response_time_ms === 'number' ? agent.response_time_ms : 0;

                                return (
                                  <div
                                    key={idx}
                                    className={cn(
                                      'flex items-center justify-between gap-4',
                                      'py-1 px-2',
                                      'bg-accent-50 dark:bg-accent-900',
                                      'rounded'
                                    )}
                                  >
                                    <span className="font-medium">{agentName}</span>
                                    <div className="flex items-center gap-3 text-[11px]">
                                      <span>${cost.toFixed(6)}</span>
                                      <span className="text-accent-400 dark:text-accent-500">
                                        {inputTokens.toLocaleString()} {t('chat.tokensIn')} / {outputTokens.toLocaleString()} {t('chat.tokensOut')}
                                      </span>
                                      {cacheTokens > 0 && (
                                        <span className="text-accent-600 dark:text-accent-400">
                                          {cacheTokens.toLocaleString()} {t('chat.cached')}
                                        </span>
                                      )}
                                      {responseTime > 0 && (
                                        <span className="text-accent-500 dark:text-accent-400 flex items-center gap-0.5">
                                          <Clock size={10} />
                                          {(responseTime / 1000).toFixed(2)}s
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </details>
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
                        {t('chat.responseTime')}: {(responseDurationMs / 1000).toFixed(2)}s
                      </div>
                    )}

                    {/* Tool usage count */}
                    {message.toolCalls && message.toolCalls.length > 0 && (
                      <div className={cn(
                        'text-xs',
                        'text-accent-600 dark:text-accent-400'
                      )}>
                        {t('chat.toolsUsed')}: {message.toolCalls.length}
                      </div>
                    )}
                  </div>
                </details>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
