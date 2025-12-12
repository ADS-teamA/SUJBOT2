/**
 * CitationLink Component
 *
 * Renders a clickable citation badge that displays document info and opens PDF viewer.
 * Uses the CitationContext to fetch metadata and manage PDF viewer state.
 *
 * Display format: [BZ_VR1:12] (document:page)
 * Tooltip shows: section path or title
 */

import { useEffect } from 'react';
import { FileText, Loader2 } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';
import { useCitationContext } from '../../contexts/CitationContext';
import { formatCitationShort, formatCitationTooltip } from '../../utils/citations';

interface CitationLinkProps {
  chunkId: string;
}

export function CitationLink({ chunkId }: CitationLinkProps) {
  const { citationCache, fetchCitationMetadata, openPdf, loadingIds } = useCitationContext();

  // Fetch metadata on mount if not cached
  useEffect(() => {
    if (!citationCache.has(chunkId)) {
      fetchCitationMetadata([chunkId]);
    }
  }, [chunkId, citationCache, fetchCitationMetadata]);

  const metadata = citationCache.get(chunkId);

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
  const tooltipText = formatCitationTooltip(metadata.sectionPath, metadata.sectionTitle);

  // PDF not available - show non-clickable badge
  if (!metadata.pdfAvailable) {
    return (
      <span
        className={cn(
          'inline-flex items-center gap-1 px-1.5 py-0.5 rounded',
          'bg-accent-100 dark:bg-accent-800',
          'text-accent-600 dark:text-accent-400',
          'text-xs font-mono'
        )}
        title={tooltipText}
      >
        <FileText size={10} className="opacity-50" />
        {displayText}
      </span>
    );
  }

  // Clickable citation with PDF available
  return (
    <button
      onClick={handleClick}
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
      title={tooltipText}
    >
      <FileText size={10} />
      {displayText}
    </button>
  );
}
