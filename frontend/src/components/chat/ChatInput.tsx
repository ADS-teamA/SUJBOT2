/**
 * ChatInput Component - Message input textarea with send/stop button
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Square, FileText, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { useAuth } from '../../contexts/AuthContext';
import { apiService, type SpendingInfo } from '../../services/api';
import type { TextSelection } from '../../types';

interface ChatInputProps {
  onSend: (message: string) => void;
  onCancel?: () => void;  // Cancel streaming
  isStreaming: boolean;   // Whether currently streaming
  disabled: boolean;      // Disabled for other reasons (not streaming)
  refreshSpendingTrigger?: number; // Increment to refresh spending data
  selectedText?: TextSelection | null;  // Selected text from PDF
  onClearSelection?: () => void;  // Clear selection callback
}

export function ChatInput({ onSend, onCancel, isStreaming, disabled, refreshSpendingTrigger, selectedText, onClearSelection }: ChatInputProps) {
  const { t } = useTranslation();
  const { user } = useAuth();

  // Calculate line count for selection (filter out empty lines)
  const selectionLineCount = selectedText
    ? selectedText.text.split('\n').filter(line => line.trim()).length || 1
    : 0;
  const [message, setMessage] = useState('');
  const [spending, setSpending] = useState<SpendingInfo | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Backend limit: 50,000 characters (see backend/models.py ChatRequest)
  const MAX_MESSAGE_LENGTH = 50000;
  const isMessageTooLong = message.length > MAX_MESSAGE_LENGTH;

  // Calculate spending status for color coding
  const spendingPercentage = spending
    ? Math.min((spending.total_spent_czk / spending.spending_limit_czk) * 100, 100)
    : 0;
  const isBlocked = spending
    ? spending.total_spent_czk >= spending.spending_limit_czk
    : false;

  // Fetch spending data with rate limit protection
  const fetchSpending = useCallback(async () => {
    try {
      const data = await apiService.getSpending();
      setSpending(data);
    } catch (error) {
      // Log 429 (rate limit) but don't show error to user - will retry on next trigger
      if (error instanceof Error && error.message.includes('429')) {
        console.warn('Rate limited when fetching spending, will retry on next trigger');
        return;
      }
      console.error('Failed to fetch spending:', error);
    }
  }, []);

  // Fetch spending on mount and when trigger changes
  // Use a small delay to prevent simultaneous fetches from multiple ChatInput instances
  useEffect(() => {
    const delay = Math.random() * 100; // Random 0-100ms delay to stagger requests
    const timeoutId = setTimeout(fetchSpending, delay);
    return () => clearTimeout(timeoutId);
  }, [fetchSpending, refreshSpendingTrigger]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (message.trim() && !disabled && !isStreaming && !isMessageTooLong) {
      onSend(message.trim());
      setMessage('');
    }
  };

  const handleCancel = () => {
    if (isStreaming && onCancel) {
      onCancel();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="p-6">
      <div className="max-w-4xl mx-auto">
        <div
          className={cn(
            'flex gap-3 p-2',
            'bg-white dark:bg-accent-900',
            'border border-accent-200 dark:border-accent-800',
            'rounded-2xl',
            'shadow-lg',
            'transition-all duration-300',
            'hover:shadow-xl',
            !disabled && 'hover:border-accent-300 dark:hover:border-accent-700'
          )}
        >
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t('chat.placeholder')}
            disabled={disabled || isStreaming}
            className={cn(
              'flex-1 resize-none px-4 py-3',
              'bg-transparent',
              'text-accent-900 dark:text-accent-100',
              'placeholder:text-accent-400 dark:placeholder:text-accent-600',
              'focus:outline-none',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'scrollbar-hide overflow-y-auto'
            )}
            rows={1}
          />
          {isStreaming ? (
            /* Stop button - shown during streaming */
            <button
              type="button"
              onClick={handleCancel}
              className={cn(
                'flex-shrink-0',
                'w-10 h-10 rounded-xl',
                'bg-red-600 dark:bg-red-500',
                'text-white',
                'hover:bg-red-700 dark:hover:bg-red-600',
                'hover:scale-105 active:scale-95',
                'transition-all duration-200',
                'flex items-center justify-center',
                'shadow-md hover:shadow-lg'
              )}
              title={t('chat.stop')}
            >
              <Square size={16} fill="currentColor" />
            </button>
          ) : (
            /* Send button - shown when not streaming */
            <button
              type="submit"
              disabled={disabled || !message.trim() || isMessageTooLong}
              className={cn(
                'flex-shrink-0',
                'w-10 h-10 rounded-xl',
                'bg-accent-900 dark:bg-accent-100',
                'text-accent-50 dark:text-accent-900',
                'hover:bg-accent-800 dark:hover:bg-accent-200',
                'hover:scale-105 active:scale-95',
                'disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100',
                'transition-all duration-200',
                'flex items-center justify-center',
                'shadow-md hover:shadow-lg'
              )}
              title={
                disabled
                  ? t('chat.processing')
                  : isMessageTooLong
                  ? `${t('chat.messageTooLong')} (${message.length.toLocaleString()}/${MAX_MESSAGE_LENGTH.toLocaleString()})`
                  : t('chat.placeholder')
              }
            >
              <Send size={18} />
            </button>
          )}
        </div>
        {/* Bottom row: selection chip (left) + user info (right) */}
        <div className="mt-2 flex justify-between items-center text-xs">
          {/* Selection chip - left side */}
          {selectedText && onClearSelection ? (
            <div
              className={cn(
                'inline-flex items-center gap-1.5 px-2 py-0.5',
                'bg-gray-100 dark:bg-gray-800',
                'text-gray-600 dark:text-gray-400',
                'rounded-full',
                'border border-gray-200 dark:border-gray-700'
              )}
              style={{ animation: 'chipIn 0.15s ease-out' }}
            >
              <FileText size={10} className="text-gray-400 dark:text-gray-500" />
              <span>{t('selection.lines', { count: selectionLineCount })}</span>
              <span className="text-gray-300 dark:text-gray-600">•</span>
              <span className="truncate max-w-[120px]" title={selectedText.documentName}>
                {selectedText.documentName.replace(/_/g, ' ')}
              </span>
              <button
                type="button"
                onClick={onClearSelection}
                className={cn(
                  'p-0.5 rounded-full -mr-0.5',
                  'text-gray-400 dark:text-gray-500',
                  'hover:bg-gray-200 dark:hover:bg-gray-700',
                  'hover:text-gray-600 dark:hover:text-gray-300',
                  'transition-colors duration-150'
                )}
                title={t('selection.clearSelection')}
                aria-label={t('selection.clearSelection')}
              >
                <X size={10} />
              </button>
            </div>
          ) : (
            <div />
          )}

          {/* User email + spending - right side */}
          <div className="text-right flex-shrink-0">
            {user && (
              <div className="text-accent-500 dark:text-accent-500 mb-0.5">
                {user.email}
              </div>
            )}
            {spending && (
              <div
                className={cn(
                  'font-medium transition-colors duration-200',
                  isBlocked
                    ? 'text-red-600 dark:text-red-400'
                    : spendingPercentage >= 90
                      ? 'text-red-600 dark:text-red-400'
                      : spendingPercentage >= 70
                        ? 'text-amber-600 dark:text-amber-400'
                        : 'text-green-600 dark:text-green-400'
                )}
              >
                {spending.total_spent_czk.toFixed(2)} / {spending.spending_limit_czk.toFixed(2)} Kč
                {isBlocked && (
                  <span className="ml-2 px-1.5 py-0.5 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 rounded text-[10px]">
                    {t('chat.blocked')}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Animation keyframes for selection chip */}
        <style>{`
          @keyframes chipIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
          }
        `}</style>
      </div>
    </form>
  );
}
