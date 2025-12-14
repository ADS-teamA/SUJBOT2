/**
 * SelectionIndicator Component
 *
 * Displays a minimalist pill/chip showing selected text info from PDF.
 * Shows: line count, document name, and close button.
 *
 * Design: Neutral gray oval chip, centered above chat input.
 */

import { X, FileText } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import type { TextSelection } from '../../types';

interface SelectionIndicatorProps {
  selection: TextSelection;
  onClear: () => void;
}

export function SelectionIndicator({ selection, onClear }: SelectionIndicatorProps) {
  const { t } = useTranslation();

  // Calculate line count (filter out empty lines)
  const lineCount = selection.text
    .split('\n')
    .filter(line => line.trim()).length || 1;

  // Format document name (replace underscores with spaces)
  const documentName = selection.documentName.replace(/_/g, ' ');

  return (
    <div className="max-w-4xl mx-auto px-6 pt-1 pb-2">
      <div
        className={cn(
          'inline-flex items-center gap-2 px-3 py-1.5',
          'bg-gray-100 dark:bg-gray-800',
          'text-gray-700 dark:text-gray-300',
          'rounded-full text-sm',
          'shadow-sm border border-gray-200 dark:border-gray-700'
        )}
        style={{
          animation: 'chipIn 0.2s ease-out',
        }}
      >
        {/* Document icon */}
        <FileText size={14} className="text-gray-500 dark:text-gray-400 flex-shrink-0" />

        {/* Line count */}
        <span className="font-medium">
          {t('selection.lines', { count: lineCount })}
        </span>

        {/* Separator */}
        <span className="text-gray-400 dark:text-gray-500">•</span>

        {/* Document name */}
        <span className="truncate max-w-[200px]" title={documentName}>
          {documentName}
        </span>

        {/* Close button */}
        <button
          onClick={onClear}
          className={cn(
            'ml-1 p-0.5 rounded-full',
            'text-gray-500 dark:text-gray-400',
            'hover:bg-gray-200 dark:hover:bg-gray-700',
            'hover:text-gray-700 dark:hover:text-gray-200',
            'transition-colors duration-150'
          )}
          title={t('selection.clearSelection', 'Clear selection')}
          aria-label={t('selection.clearSelection', 'Clear selection')}
        >
          <X size={14} />
        </button>
      </div>

      {/* Animation keyframes */}
      <style>{`
        @keyframes chipIn {
          from {
            opacity: 0;
            transform: scale(0.95) translateY(4px);
          }
          to {
            opacity: 1;
            transform: scale(1) translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
