/**
 * ClarificationModal Component - HITL Clarification Dialog
 *
 * Displays when the multi-agent system detects a poorly-specified query
 * and needs user clarification to improve retrieval quality.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { X, CheckCircle2, HelpCircle } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';
import { useFadeIn } from '../../design-system/animations/hooks/useFadeIn';
import type { ClarificationData } from '../../types';

interface ClarificationModalProps {
  isOpen: boolean;
  clarificationData: ClarificationData | null;
  onSubmit: (response: string) => void;
  onCancel: () => void;
  disabled?: boolean;
}

export function ClarificationModal({
  isOpen,
  clarificationData,
  onSubmit,
  onCancel,
  disabled = false,
}: ClarificationModalProps) {
  const [userResponse, setUserResponse] = useState('');
  const [timeRemaining, setTimeRemaining] = useState<number | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { style: fadeStyle } = useFadeIn({ duration: 'normal' });

  // Initialize timer when modal opens
  useEffect(() => {
    if (!isOpen || !clarificationData) {
      setTimeRemaining(null);
      return;
    }

    // Set initial time
    setTimeRemaining(clarificationData.timeout_seconds);

    // Countdown timer
    const interval = setInterval(() => {
      setTimeRemaining((prev) => {
        if (prev === null || prev <= 1) {
          clearInterval(interval);
          // Auto-submit with empty response on timeout
          onCancel();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [isOpen, clarificationData, onCancel]);

  // Focus textarea when modal opens
  useEffect(() => {
    if (isOpen && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isOpen]);

  // Reset response when modal closes
  useEffect(() => {
    if (!isOpen) {
      setUserResponse('');
    }
  }, [isOpen]);

  const handleSubmit = useCallback(() => {
    if (!userResponse.trim()) {
      return;
    }
    onSubmit(userResponse.trim());
    setUserResponse('');
  }, [userResponse, onSubmit]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      // Submit on Ctrl+Enter or Cmd+Enter
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  if (!isOpen || !clarificationData) {
    return null;
  }

  const { questions, quality_metrics, original_query } = clarificationData;

  // Format time remaining
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Get quality status color
  const getQualityColor = (score: number) => {
    if (score >= 0.7) return 'text-green-600 dark:text-green-400';
    if (score >= 0.5) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className={cn(
          'fixed inset-0 z-50',
          'bg-black/50 dark:bg-black/70',
          'backdrop-blur-sm',
          'transition-opacity duration-300'
        )}
        onClick={disabled ? undefined : onCancel}
        style={fadeStyle}
      />

      {/* Modal */}
      <div
        className={cn(
          'fixed inset-0 z-50',
          'flex items-center justify-center',
          'p-4'
        )}
      >
        <div
          className={cn(
            'bg-white dark:bg-accent-900',
            'rounded-lg shadow-2xl',
            'w-full max-w-2xl max-h-[90vh]',
            'overflow-hidden',
            'border border-accent-200 dark:border-accent-700',
            'transform transition-all duration-300'
          )}
          style={fadeStyle}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div
            className={cn(
              'flex items-center justify-between',
              'px-6 py-4',
              'border-b border-accent-200 dark:border-accent-700',
              'bg-accent-50 dark:bg-accent-800'
            )}
          >
            <div className="flex items-center gap-3">
              <HelpCircle className="text-blue-600 dark:text-blue-400" size={24} />
              <div>
                <h2 className="text-xl font-semibold text-accent-900 dark:text-accent-100">
                  Query Clarification Needed
                </h2>
                <p className="text-sm text-accent-600 dark:text-accent-400">
                  Help improve search quality by providing more details
                </p>
              </div>
            </div>
            <button
              onClick={onCancel}
              disabled={disabled}
              className={cn(
                'p-2 rounded-lg',
                'text-accent-500 hover:text-accent-700',
                'dark:text-accent-400 dark:hover:text-accent-200',
                'hover:bg-accent-100 dark:hover:bg-accent-700',
                'transition-colors',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
              aria-label="Close"
            >
              <X size={20} />
            </button>
          </div>

          {/* Content */}
          <div className="overflow-y-auto max-h-[calc(90vh-200px)]">
            <div className="px-6 py-4 space-y-4">
              {/* Timer */}
              {timeRemaining !== null && (
                <div
                  className={cn(
                    'flex items-center justify-between',
                    'px-4 py-2 rounded-lg',
                    'bg-yellow-50 dark:bg-yellow-900/20',
                    'border border-yellow-200 dark:border-yellow-800'
                  )}
                >
                  <span className="text-sm text-yellow-800 dark:text-yellow-200">
                    Time remaining to respond:
                  </span>
                  <span className="text-lg font-mono font-semibold text-yellow-900 dark:text-yellow-100">
                    {formatTime(timeRemaining)}
                  </span>
                </div>
              )}

              {/* Original Query */}
              <div>
                <h3 className="text-sm font-semibold text-accent-700 dark:text-accent-300 mb-2">
                  Your Original Query:
                </h3>
                <div
                  className={cn(
                    'px-4 py-3 rounded-lg',
                    'bg-accent-100 dark:bg-accent-800',
                    'text-accent-900 dark:text-accent-100',
                    'text-sm italic'
                  )}
                >
                  "{original_query}"
                </div>
              </div>

              {/* Quality Metrics - Hidden for cleaner UX */}
              {/* Detection metrics are still collected but not shown to user */}

              {/* Clarification Questions */}
              <div>
                <h3 className="text-sm font-semibold text-accent-700 dark:text-accent-300 mb-3">
                  Please answer these questions to improve your search:
                </h3>
                <div className="space-y-3">
                  {questions.map((question, index) => (
                    <div
                      key={question.id}
                      className={cn(
                        'flex gap-3',
                        'px-4 py-3 rounded-lg',
                        'bg-blue-50 dark:bg-blue-900/20',
                        'border border-blue-200 dark:border-blue-800'
                      )}
                    >
                      <span className="text-blue-600 dark:text-blue-400 font-semibold flex-shrink-0">
                        {index + 1}.
                      </span>
                      <span className="text-accent-900 dark:text-accent-100 text-sm">
                        {question.text}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* User Response Textarea */}
              <div>
                <label
                  htmlFor="clarification-response"
                  className="block text-sm font-semibold text-accent-700 dark:text-accent-300 mb-2"
                >
                  Your Response:
                </label>
                <textarea
                  ref={textareaRef}
                  id="clarification-response"
                  value={userResponse}
                  onChange={(e) => setUserResponse(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={disabled}
                  placeholder="Please provide any clarifying details that would help refine your search..."
                  className={cn(
                    'w-full px-4 py-3 rounded-lg',
                    'bg-white dark:bg-accent-800',
                    'border border-accent-300 dark:border-accent-600',
                    'text-accent-900 dark:text-accent-100',
                    'placeholder-accent-400 dark:placeholder-accent-500',
                    'focus:outline-none focus:ring-2',
                    'focus:ring-blue-500 dark:focus:ring-blue-400',
                    'focus:border-transparent',
                    'resize-none',
                    'text-sm',
                    disabled && 'opacity-50 cursor-not-allowed'
                  )}
                  rows={6}
                />
                <p className="mt-2 text-xs text-accent-500 dark:text-accent-400">
                  Tip: Press <kbd className="px-1 py-0.5 bg-accent-200 dark:bg-accent-700 rounded">Ctrl</kbd> +{' '}
                  <kbd className="px-1 py-0.5 bg-accent-200 dark:bg-accent-700 rounded">Enter</kbd> to submit
                </p>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div
            className={cn(
              'flex items-center justify-end gap-3',
              'px-6 py-4',
              'border-t border-accent-200 dark:border-accent-700',
              'bg-accent-50 dark:bg-accent-800'
            )}
          >
            <button
              onClick={onCancel}
              disabled={disabled}
              className={cn(
                'px-4 py-2 rounded-lg',
                'text-accent-700 dark:text-accent-300',
                'hover:bg-accent-200 dark:hover:bg-accent-700',
                'transition-colors',
                'text-sm font-medium',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              Continue Without Clarification
            </button>
            <button
              onClick={handleSubmit}
              disabled={disabled || !userResponse.trim()}
              className={cn(
                'px-6 py-2 rounded-lg',
                'bg-blue-600 hover:bg-blue-700',
                'dark:bg-blue-500 dark:hover:bg-blue-600',
                'text-white',
                'transition-colors',
                'text-sm font-semibold',
                'flex items-center gap-2',
                (disabled || !userResponse.trim()) && 'opacity-50 cursor-not-allowed'
              )}
            >
              <CheckCircle2 size={16} />
              Submit Clarification
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
