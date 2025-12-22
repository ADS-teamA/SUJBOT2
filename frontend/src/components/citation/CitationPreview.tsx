/**
 * CitationPreview Component
 *
 * Floating preview popover shown on citation hover.
 * Auto-positions above/below based on available viewport space.
 *
 * Features:
 * - Fade in/out animation (respects prefers-reduced-motion)
 * - Automatic flip positioning
 * - Max-height with scroll for very long content
 * - Dark mode support
 */

import { useEffect, useRef, useState } from 'react';
import { FileText } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { usePopoverPosition } from '../../hooks/usePopoverPosition';
import { useMediaQuery } from '../../hooks/useMediaQuery';
import { timing } from '../../design-system/tokens/timing';
import type { CitationMetadata } from '../../types';

interface CitationPreviewProps {
  metadata: CitationMetadata;
  anchorRect: DOMRect | null;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

// Popover dimensions
const POPOVER_WIDTH = 320;
const POPOVER_MAX_HEIGHT = 300;
const ESTIMATED_HEIGHT = 200; // Initial estimate for positioning

export function CitationPreview({
  metadata,
  anchorRect,
  onMouseEnter,
  onMouseLeave,
}: CitationPreviewProps) {
  const { t } = useTranslation();
  const popoverRef = useRef<HTMLDivElement>(null);
  const [actualHeight, setActualHeight] = useState(ESTIMATED_HEIGHT);
  const [isVisible, setIsVisible] = useState(false);

  // Respect accessibility preferences
  const prefersReducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)');

  // Calculate position
  const position = usePopoverPosition({
    anchorRect,
    popoverHeight: actualHeight,
    popoverWidth: POPOVER_WIDTH,
    gap: 8,
  });

  // Measure actual height after render
  useEffect(() => {
    if (popoverRef.current) {
      const height = Math.min(popoverRef.current.scrollHeight, POPOVER_MAX_HEIGHT);
      setActualHeight(height);
    }
  }, [metadata.content]);

  // Fade in animation
  useEffect(() => {
    // Small delay for DOM to settle before animation
    const timer = setTimeout(() => setIsVisible(true), 10);
    return () => clearTimeout(timer);
  }, []);

  if (!position) return null;

  const displayPath = metadata.sectionPath || metadata.hierarchicalPath || metadata.documentName;
  const hasContent = metadata.content && metadata.content.trim().length > 0;

  return (
    <div
      ref={popoverRef}
      role="tooltip"
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      className={cn(
        'fixed z-50',
        'bg-white dark:bg-accent-900',
        'border border-accent-200 dark:border-accent-700',
        'rounded-lg shadow-xl',
        'text-sm',
        'overflow-hidden'
      )}
      style={{
        top: position.top,
        left: position.left,
        width: POPOVER_WIDTH,
        maxHeight: POPOVER_MAX_HEIGHT,
        opacity: isVisible ? 1 : 0,
        transform: isVisible
          ? 'translateY(0)'
          : position.placement === 'bottom'
            ? 'translateY(-4px)'
            : 'translateY(4px)',
        transition: prefersReducedMotion
          ? 'none'
          : `opacity ${timing.durations.fast} ${timing.easings.easeOut}, transform ${timing.durations.fast} ${timing.easings.easeOut}`,
      }}
    >
      {/* Arrow */}
      <div
        className={cn(
          'absolute w-3 h-3',
          'bg-white dark:bg-accent-900',
          'border-accent-200 dark:border-accent-700',
          'transform rotate-45',
          position.placement === 'bottom'
            ? '-top-1.5 border-l border-t'
            : '-bottom-1.5 border-r border-b'
        )}
        style={{ left: position.arrowLeft - 6 }}
      />

      {/* Header */}
      <div
        className={cn(
          'flex items-center gap-2 px-3 py-2',
          'border-b border-accent-100 dark:border-accent-800',
          'bg-accent-50 dark:bg-accent-800/50',
          'font-medium text-accent-900 dark:text-accent-100'
        )}
      >
        <FileText size={14} className="flex-shrink-0 text-blue-600 dark:text-blue-400" />
        <span className="truncate text-xs">{displayPath}</span>
      </div>

      {/* Page info */}
      {metadata.pageNumber && (
        <div
          className={cn(
            'px-3 py-1.5',
            'border-b border-accent-100 dark:border-accent-800',
            'text-xs text-accent-500 dark:text-accent-400'
          )}
        >
          {t('citation.page', 'Page')} {metadata.pageNumber}
        </div>
      )}

      {/* Content */}
      <div
        className={cn(
          'px-3 py-2',
          'text-accent-700 dark:text-accent-300',
          'leading-relaxed',
          'overflow-y-auto'
        )}
        style={{ maxHeight: POPOVER_MAX_HEIGHT - 80 }} // Account for header
      >
        {hasContent ? (
          <p className="whitespace-pre-wrap break-words">{metadata.content}</p>
        ) : (
          <p className="text-accent-400 dark:text-accent-500 italic">
            {t('citation.noContent', 'No preview available')}
          </p>
        )}
      </div>

      {/* Footer hint */}
      {metadata.pdfAvailable && (
        <div
          className={cn(
            'px-3 py-1.5',
            'border-t border-accent-100 dark:border-accent-800',
            'text-xs text-accent-400 dark:text-accent-500',
            'bg-accent-50/50 dark:bg-accent-800/30'
          )}
        >
          {t('citation.clickToView', 'Click to view in PDF')}
        </div>
      )}
    </div>
  );
}
