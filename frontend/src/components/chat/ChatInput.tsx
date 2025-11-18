/**
 * ChatInput Component - Message input textarea with send button
 */

import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (message.trim() && !disabled) {
      onSend(message.trim());
      setMessage('');
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
            placeholder="Ask about legal or technical documents..."
            disabled={disabled}
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
          <button
            type="submit"
            disabled={disabled || !message.trim()}
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
            title={disabled ? 'Processing...' : 'Send message'}
          >
            {disabled ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Send size={18} />
            )}
          </button>
        </div>
        {message.length > 0 && (
          <div
            className={cn(
              'mt-2 px-2 text-xs',
              'text-accent-500 dark:text-accent-500',
              'text-right'
            )}
          >
            {message.length} characters
          </div>
        )}
      </div>
    </form>
  );
}
