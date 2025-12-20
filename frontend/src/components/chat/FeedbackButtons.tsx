/**
 * FeedbackButtons Component - User feedback for assistant messages
 *
 * Features:
 * - Thumbs up/down rating
 * - Optional comment with modal
 * - Sends to PostgreSQL + LangSmith
 * - Disabled during streaming
 */

import { useState } from 'react';
import { ThumbsUp, ThumbsDown, MessageSquare, X, Send, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { apiService } from '../../services/api';
import type { MessageFeedback } from '../../types';

interface FeedbackButtonsProps {
  /** Database message ID (required for API call) */
  dbMessageId?: number;
  /** LangSmith run/trace ID for correlation */
  runId?: string;
  /** Existing feedback (if already submitted) */
  existingFeedback?: MessageFeedback;
  /** Whether buttons are disabled (e.g., during streaming) */
  disabled?: boolean;
  /** Callback when feedback is submitted */
  onFeedbackSubmit?: (feedback: MessageFeedback) => void;
}

export function FeedbackButtons({
  dbMessageId,
  runId,
  existingFeedback,
  disabled = false,
  onFeedbackSubmit,
}: FeedbackButtonsProps) {
  const { t } = useTranslation();

  // Local state
  const [feedback, setFeedback] = useState<MessageFeedback | undefined>(existingFeedback);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showCommentModal, setShowCommentModal] = useState(false);
  const [comment, setComment] = useState('');
  const [pendingScore, setPendingScore] = useState<1 | -1 | null>(null);

  // If no database message ID, can't submit feedback
  if (!dbMessageId) {
    return null;
  }

  // If feedback already submitted, show it as read-only
  if (feedback) {
    return (
      <div className="flex items-center gap-1 mt-2">
        <div
          className={cn(
            'flex items-center gap-1 px-2 py-1 rounded-full text-xs',
            feedback.score === 1
              ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
              : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
          )}
        >
          {feedback.score === 1 ? (
            <ThumbsUp className="w-3 h-3" />
          ) : (
            <ThumbsDown className="w-3 h-3" />
          )}
          <span>{t('feedback.submitted')}</span>
        </div>
        {feedback.comment && (
          <div className="text-xs text-gray-500 dark:text-gray-400 italic ml-2 max-w-[200px] truncate">
            "{feedback.comment}"
          </div>
        )}
      </div>
    );
  }

  const handleScore = async (score: 1 | -1) => {
    if (isSubmitting || disabled) return;

    setIsSubmitting(true);
    try {
      await apiService.submitFeedback(dbMessageId, score, runId);
      const newFeedback: MessageFeedback = { score };
      setFeedback(newFeedback);
      onFeedbackSubmit?.(newFeedback);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      // Could show a toast here
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCommentClick = (score: 1 | -1) => {
    setPendingScore(score);
    setShowCommentModal(true);
  };

  const handleCommentSubmit = async () => {
    if (!pendingScore || isSubmitting) return;

    setIsSubmitting(true);
    try {
      await apiService.submitFeedback(dbMessageId, pendingScore, runId, comment.trim() || undefined);
      const newFeedback: MessageFeedback = {
        score: pendingScore,
        comment: comment.trim() || undefined,
      };
      setFeedback(newFeedback);
      onFeedbackSubmit?.(newFeedback);
      setShowCommentModal(false);
      setComment('');
      setPendingScore(null);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCommentCancel = () => {
    setShowCommentModal(false);
    setComment('');
    setPendingScore(null);
  };

  return (
    <>
      <div className="flex items-center gap-1 mt-2">
        {/* Thumbs up */}
        <button
          onClick={() => handleScore(1)}
          disabled={isSubmitting || disabled}
          className={cn(
            'p-1.5 rounded-lg transition-colors',
            'text-gray-400 hover:text-green-600 hover:bg-green-50',
            'dark:text-gray-500 dark:hover:text-green-400 dark:hover:bg-green-900/20',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            isSubmitting && 'animate-pulse'
          )}
          title={t('feedback.helpful')}
        >
          <ThumbsUp className="w-4 h-4" />
        </button>

        {/* Thumbs down */}
        <button
          onClick={() => handleScore(-1)}
          disabled={isSubmitting || disabled}
          className={cn(
            'p-1.5 rounded-lg transition-colors',
            'text-gray-400 hover:text-red-600 hover:bg-red-50',
            'dark:text-gray-500 dark:hover:text-red-400 dark:hover:bg-red-900/20',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            isSubmitting && 'animate-pulse'
          )}
          title={t('feedback.notHelpful')}
        >
          <ThumbsDown className="w-4 h-4" />
        </button>

        {/* Comment button */}
        <button
          onClick={() => handleCommentClick(1)}
          disabled={isSubmitting || disabled}
          className={cn(
            'p-1.5 rounded-lg transition-colors',
            'text-gray-400 hover:text-blue-600 hover:bg-blue-50',
            'dark:text-gray-500 dark:hover:text-blue-400 dark:hover:bg-blue-900/20',
            'disabled:opacity-50 disabled:cursor-not-allowed'
          )}
          title={t('feedback.addComment')}
        >
          <MessageSquare className="w-4 h-4" />
        </button>
      </div>

      {/* Comment Modal */}
      {showCommentModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-4 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                {t('feedback.addComment')}
              </h3>
              <button
                onClick={handleCommentCancel}
                className="p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>

            {/* Score selection */}
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {t('feedback.rating')}:
              </span>
              <button
                onClick={() => setPendingScore(1)}
                className={cn(
                  'p-2 rounded-lg transition-colors',
                  pendingScore === 1
                    ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                    : 'text-gray-400 hover:text-green-600 hover:bg-green-50 dark:hover:bg-green-900/20'
                )}
              >
                <ThumbsUp className="w-5 h-5" />
              </button>
              <button
                onClick={() => setPendingScore(-1)}
                className={cn(
                  'p-2 rounded-lg transition-colors',
                  pendingScore === -1
                    ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                    : 'text-gray-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20'
                )}
              >
                <ThumbsDown className="w-5 h-5" />
              </button>
            </div>

            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder={t('feedback.commentPlaceholder')}
              className={cn(
                'w-full h-24 px-3 py-2 rounded-lg border resize-none',
                'bg-gray-50 dark:bg-gray-900',
                'border-gray-200 dark:border-gray-700',
                'text-gray-900 dark:text-white',
                'placeholder:text-gray-400 dark:placeholder:text-gray-500',
                'focus:outline-none focus:ring-2 focus:ring-blue-500'
              )}
              maxLength={2000}
            />

            <div className="flex items-center justify-between mt-3">
              <span className="text-xs text-gray-400">
                {comment.length}/2000
              </span>
              <div className="flex gap-2">
                <button
                  onClick={handleCommentCancel}
                  className={cn(
                    'px-4 py-2 rounded-lg text-sm font-medium',
                    'text-gray-600 dark:text-gray-300',
                    'hover:bg-gray-100 dark:hover:bg-gray-700'
                  )}
                >
                  {t('feedback.cancel')}
                </button>
                <button
                  onClick={handleCommentSubmit}
                  disabled={!pendingScore || isSubmitting}
                  className={cn(
                    'px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2',
                    'bg-blue-600 text-white hover:bg-blue-700',
                    'disabled:opacity-50 disabled:cursor-not-allowed'
                  )}
                >
                  {isSubmitting ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                  {t('feedback.submit')}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
