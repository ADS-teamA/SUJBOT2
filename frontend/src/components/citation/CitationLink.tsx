/**
 * CitationLink Component
 *
 * Renders a clickable citation badge that displays document info and opens PDF viewer.
 * Uses the CitationContext to fetch metadata and manage PDF viewer state.
 *
 * Display format: [BZ_VR1:12] (document:page)
 * Hover shows: rich preview with chunk content
 */

import { useEffect, useState, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { FileText, Loader2 } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';
import { useCitationContext } from '../../contexts/CitationContext';
import { formatCitationShort } from '../../utils/citations';
import { CitationPreview } from './CitationPreview';

interface CitationLinkProps {
  chunkId: string;
}

// Hover delay constants (ms)
const HOVER_ENTER_DELAY = 200; // Delay before showing preview
const HOVER_LEAVE_DELAY = 100; // Delay before hiding (allows moving to preview)

export function CitationLink({ chunkId }: CitationLinkProps) {
  const { citationCache, fetchCitationMetadata, openPdf, loadingIds } = useCitationContext();

  // Hover state for preview
  const [isHovered, setIsHovered] = useState(false);
  const [anchorRect, setAnchorRect] = useState<DOMRect | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const spanRef = useRef<HTMLSpanElement>(null);
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // Fetch metadata on mount if not cached
  useEffect(() => {
    if (!citationCache.has(chunkId)) {
      fetchCitationMetadata([chunkId]);
    }
  }, [chunkId, citationCache, fetchCitationMetadata]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
    };
  }, []);

  const metadata = citationCache.get(chunkId);

  // Hover handlers with delay
  const handleMouseEnter = useCallback((ref: React.RefObject<HTMLElement | null>) => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    hoverTimeoutRef.current = setTimeout(() => {
      setIsHovered(true);
      if (ref.current) {
        setAnchorRect(ref.current.getBoundingClientRect());
      }
    }, HOVER_ENTER_DELAY);
  }, []);

  const handleMouseLeave = useCallback(() => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    hoverTimeoutRef.current = setTimeout(() => {
      setIsHovered(false);
    }, HOVER_LEAVE_DELAY);
  }, []);

  // Keep preview open when hovering the preview itself
  const handlePreviewMouseEnter = useCallback(() => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
  }, []);

  const handlePreviewMouseLeave = useCallback(() => {
    setIsHovered(false);
  }, []);

  // Handle click - open PDF side panel
  const handleClick = () => {
    if (metadata?.pdfAvailable) {
      openPdf(metadata.documentId, metadata.documentName, metadata.pageNumber ?? 1, chunkId);
    }
  };

  // Loading state - only show spinner if THIS specific citation is being loaded
  if (!metadata && loadingIds.has(chunkId)) {
    return (
      <span
        className={cn(
          'inline-flex items-center gap-1 px-1.5 py-0.5 rounded',
          'bg-accent-100 dark:bg-accent-800',
          'text-accent-500 dark:text-accent-400',
          'text-xs font-mono'
        )}
      >
        <Loader2 size={10} className="animate-spin" />
        <span>[...]</span>
      </span>
    );
  }

  // Fallback if metadata not found
  if (!metadata) {
    return (
      <span
        className={cn(
          'inline-flex items-center gap-1 px-1.5 py-0.5 rounded',
          'bg-accent-100 dark:bg-accent-800',
          'text-accent-500 dark:text-accent-400',
          'text-xs font-mono'
        )}
        title={`Citation: ${chunkId}`}
      >
        [{chunkId}]
      </span>
    );
  }

  const displayText = formatCitationShort(metadata.documentId, metadata.pageNumber);

  // Render preview portal helper
  const renderPreview = () => {
    if (!isHovered || !metadata) return null;
    return createPortal(
      <CitationPreview
        metadata={metadata}
        anchorRect={anchorRect}
        onMouseEnter={handlePreviewMouseEnter}
        onMouseLeave={handlePreviewMouseLeave}
      />,
      document.body
    );
  };

  // PDF not available - show non-clickable badge with hover preview
  if (!metadata.pdfAvailable) {
    return (
      <>
        <span
          ref={spanRef}
          onMouseEnter={() => handleMouseEnter(spanRef)}
          onMouseLeave={handleMouseLeave}
          className={cn(
            'inline-flex items-center gap-1 px-1.5 py-0.5 rounded',
            'bg-accent-100 dark:bg-accent-800',
            'text-accent-600 dark:text-accent-400',
            'text-xs font-mono'
          )}
        >
          <FileText size={10} className="opacity-50" />
          {displayText}
        </span>
        {renderPreview()}
      </>
    );
  }

  // Clickable citation with PDF available and hover preview
  return (
    <>
      <button
        ref={buttonRef}
        onClick={handleClick}
        onMouseEnter={() => handleMouseEnter(buttonRef)}
        onMouseLeave={handleMouseLeave}
        className={cn(
          'inline-flex items-center gap-1 px-1.5 py-0.5 rounded',
          'bg-blue-100 dark:bg-blue-900/40',
          'text-blue-700 dark:text-blue-300',
          'hover:bg-blue-200 dark:hover:bg-blue-800/60',
          'hover:text-blue-800 dark:hover:text-blue-200',
          'text-xs font-mono',
          'transition-colors duration-150',
          'cursor-pointer',
          'border border-blue-200 dark:border-blue-700/50'
        )}
      >
        <FileText size={10} />
        {displayText}
      </button>
      {renderPreview()}
    </>
  );
}
