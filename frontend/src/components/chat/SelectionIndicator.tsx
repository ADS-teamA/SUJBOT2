/**
 * SelectionIndicator Component
 *
 * Displays information about selected text from PDF that will be
 * included as context in the next message to the agent.
 *
 * Shows: character count, document name, page range, and clear button.
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

  // Format page range display
  const pageDisplay = selection.pageStart === selection.pageEnd
    ? `${t('selection.page', 'page')} ${selection.pageStart}`
    : `${t('selection.pages', 'pages')} ${selection.pageStart}-${selection.pageEnd}`;

  return (
    <div
      className={cn(
        'flex items-center gap-3 px-4 py-2',
        'bg-blue-50 dark:bg-blue-900/20',
        'border-b border-blue-200 dark:border-blue-800',
        'text-sm'
      )}
    >
      {/* Icon */}
      <FileText size={16} className="text-blue-500 dark:text-blue-400 flex-shrink-0" />

      {/* Selection info */}
      <div className="flex-1 min-w-0">
        <span className="text-blue-700 dark:text-blue-300 font-medium">
          {t('selection.charactersSelected', '{{count}} characters selected', { count: selection.charCount })}
        </span>
        <span className="text-blue-600 dark:text-blue-400 mx-2">|</span>
        <span className="text-blue-600 dark:text-blue-400 truncate">
          {selection.documentName.replace(/_/g, ' ')} ({pageDisplay})
        </span>
      </div>

      {/* Clear button */}
      <button
        onClick={onClear}
        className={cn(
          'flex items-center gap-1.5 px-2 py-1 rounded',
          'text-blue-600 dark:text-blue-400',
          'hover:bg-blue-100 dark:hover:bg-blue-800/40',
          'transition-colors duration-150'
        )}
        title={t('selection.clearSelection', 'Clear selection')}
      >
        <X size={14} />
        <span className="text-xs">{t('selection.clear', 'Clear')}</span>
      </button>
    </div>
  );
}
