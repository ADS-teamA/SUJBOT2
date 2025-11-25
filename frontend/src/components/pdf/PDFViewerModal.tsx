/**
 * PDFViewerModal Component
 *
 * Fullscreen modal overlay for viewing PDF documents.
 * Uses react-pdf to render PDF pages with navigation controls.
 *
 * Features:
 * - Page navigation (prev/next, direct input)
 * - Zoom controls
 * - Keyboard navigation (arrow keys, escape to close)
 * - Initial page navigation from citation context
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import {
  X,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Loader2,
  AlertCircle,
} from 'lucide-react';
import { cn } from '../../design-system/utils/cn';

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

// API base URL from environment
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

interface PDFViewerModalProps {
  isOpen: boolean;
  documentId: string;
  initialPage?: number;
  onClose: () => void;
}

export function PDFViewerModal({
  isOpen,
  documentId,
  initialPage = 1,
  onClose,
}: PDFViewerModalProps) {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [currentPage, setCurrentPage] = useState(initialPage);
  const [scale, setScale] = useState(1.2);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [pageInputValue, setPageInputValue] = useState(String(initialPage));

  const containerRef = useRef<HTMLDivElement>(null);

  // PDF URL with authentication (needs /api/ prefix for nginx routing)
  const pdfUrl = `${API_BASE_URL}/api/documents/${documentId}/pdf`;

  // Memoize options to prevent unnecessary Document reloads
  const documentOptions = useMemo(() => ({
    withCredentials: true,
  }), []);

  // Reset state when document changes
  useEffect(() => {
    setCurrentPage(initialPage);
    setPageInputValue(String(initialPage));
    setError(null);
    setIsLoading(true);
  }, [documentId, initialPage]);

  // Handle document load success
  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
    setIsLoading(false);
    setError(null);
  }, []);

  // Handle document load error
  const onDocumentLoadError = useCallback((error: Error) => {
    console.error('PDF load error:', error);
    setError('Failed to load PDF document');
    setIsLoading(false);
  }, []);

  // Navigation handlers
  const goToPrevPage = useCallback(() => {
    setCurrentPage((prev) => Math.max(1, prev - 1));
  }, []);

  const goToNextPage = useCallback(() => {
    setCurrentPage((prev) => Math.min(numPages ?? prev, prev + 1));
  }, [numPages]);

  const goToPage = useCallback(
    (page: number) => {
      if (numPages && page >= 1 && page <= numPages) {
        setCurrentPage(page);
        setPageInputValue(String(page));
      }
    },
    [numPages]
  );

  // Update page input when currentPage changes
  useEffect(() => {
    setPageInputValue(String(currentPage));
  }, [currentPage]);

  // Handle page input change
  const handlePageInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPageInputValue(e.target.value);
  };

  // Handle page input submit
  const handlePageInputSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const page = parseInt(pageInputValue, 10);
    if (!isNaN(page)) {
      goToPage(page);
    }
  };

  // Zoom handlers
  const zoomIn = useCallback(() => {
    setScale((prev) => Math.min(3, prev + 0.2));
  }, []);

  const zoomOut = useCallback(() => {
    setScale((prev) => Math.max(0.5, prev - 0.2));
  }, []);

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          onClose();
          break;
        case 'ArrowLeft':
          goToPrevPage();
          break;
        case 'ArrowRight':
          goToNextPage();
          break;
        case '+':
        case '=':
          zoomIn();
          break;
        case '-':
          zoomOut();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, goToPrevPage, goToNextPage, zoomIn, zoomOut]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className={cn(
          'fixed inset-0 z-50',
          'bg-black/80 dark:bg-black/90',
          'backdrop-blur-sm',
          'transition-opacity duration-300'
        )}
        onClick={onClose}
      />

      {/* Modal */}
      <div
        className={cn(
          'fixed inset-0 z-50',
          'flex flex-col',
          'pointer-events-none'
        )}
      >
        {/* Header */}
        <div
          className={cn(
            'flex items-center justify-between',
            'px-4 py-3',
            'bg-white dark:bg-accent-900',
            'border-b border-accent-200 dark:border-accent-700',
            'pointer-events-auto'
          )}
        >
          {/* Document info */}
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-accent-900 dark:text-accent-100">
              {documentId.replace(/_/g, ' ')}
            </h2>
            {numPages && (
              <span className="text-sm text-accent-500 dark:text-accent-400">
                ({numPages} pages)
              </span>
            )}
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            {/* Zoom controls */}
            <div className="flex items-center gap-1 mr-4">
              <button
                onClick={zoomOut}
                disabled={scale <= 0.5}
                className={cn(
                  'p-2 rounded-lg',
                  'text-accent-600 hover:text-accent-900',
                  'dark:text-accent-400 dark:hover:text-accent-100',
                  'hover:bg-accent-100 dark:hover:bg-accent-800',
                  'disabled:opacity-50 disabled:cursor-not-allowed',
                  'transition-colors'
                )}
                title="Zoom out (-)"
              >
                <ZoomOut size={18} />
              </button>
              <span className="text-sm font-mono text-accent-600 dark:text-accent-400 min-w-[3rem] text-center">
                {Math.round(scale * 100)}%
              </span>
              <button
                onClick={zoomIn}
                disabled={scale >= 3}
                className={cn(
                  'p-2 rounded-lg',
                  'text-accent-600 hover:text-accent-900',
                  'dark:text-accent-400 dark:hover:text-accent-100',
                  'hover:bg-accent-100 dark:hover:bg-accent-800',
                  'disabled:opacity-50 disabled:cursor-not-allowed',
                  'transition-colors'
                )}
                title="Zoom in (+)"
              >
                <ZoomIn size={18} />
              </button>
            </div>

            {/* Page navigation */}
            <div className="flex items-center gap-2 mr-4">
              <button
                onClick={goToPrevPage}
                disabled={currentPage <= 1}
                className={cn(
                  'p-2 rounded-lg',
                  'text-accent-600 hover:text-accent-900',
                  'dark:text-accent-400 dark:hover:text-accent-100',
                  'hover:bg-accent-100 dark:hover:bg-accent-800',
                  'disabled:opacity-50 disabled:cursor-not-allowed',
                  'transition-colors'
                )}
                title="Previous page (←)"
              >
                <ChevronLeft size={18} />
              </button>

              <form onSubmit={handlePageInputSubmit} className="flex items-center gap-1">
                <input
                  type="text"
                  value={pageInputValue}
                  onChange={handlePageInputChange}
                  className={cn(
                    'w-12 px-2 py-1 rounded',
                    'text-center text-sm font-mono',
                    'bg-accent-100 dark:bg-accent-800',
                    'text-accent-900 dark:text-accent-100',
                    'border border-accent-300 dark:border-accent-600',
                    'focus:outline-none focus:ring-2 focus:ring-blue-500'
                  )}
                />
                <span className="text-sm text-accent-500 dark:text-accent-400">
                  / {numPages ?? '?'}
                </span>
              </form>

              <button
                onClick={goToNextPage}
                disabled={!numPages || currentPage >= numPages}
                className={cn(
                  'p-2 rounded-lg',
                  'text-accent-600 hover:text-accent-900',
                  'dark:text-accent-400 dark:hover:text-accent-100',
                  'hover:bg-accent-100 dark:hover:bg-accent-800',
                  'disabled:opacity-50 disabled:cursor-not-allowed',
                  'transition-colors'
                )}
                title="Next page (→)"
              >
                <ChevronRight size={18} />
              </button>
            </div>

            {/* Close button */}
            <button
              onClick={onClose}
              className={cn(
                'p-2 rounded-lg',
                'text-accent-500 hover:text-accent-700',
                'dark:text-accent-400 dark:hover:text-accent-200',
                'hover:bg-accent-100 dark:hover:bg-accent-800',
                'transition-colors'
              )}
              title="Close (Esc)"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* PDF Content */}
        <div
          ref={containerRef}
          className={cn(
            'flex-1 overflow-auto',
            'flex items-start justify-center',
            'p-4',
            'pointer-events-auto',
            'bg-accent-800 dark:bg-accent-950'
          )}
          onClick={(e) => e.stopPropagation()}
        >
          {error ? (
            <div
              className={cn(
                'flex flex-col items-center gap-4',
                'p-8 rounded-lg',
                'bg-red-50 dark:bg-red-900/20',
                'text-red-600 dark:text-red-400'
              )}
            >
              <AlertCircle size={48} />
              <p className="text-lg font-medium">{error}</p>
              <p className="text-sm text-accent-500 dark:text-accent-400">
                Document: {documentId}
              </p>
            </div>
          ) : (
            <Document
              file={pdfUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={
                <div className="flex items-center gap-3 text-white">
                  <Loader2 size={24} className="animate-spin" />
                  <span>Loading PDF...</span>
                </div>
              }
              options={documentOptions}
            >
              {isLoading ? null : (
                <Page
                  pageNumber={currentPage}
                  scale={scale}
                  renderTextLayer={true}
                  renderAnnotationLayer={true}
                  className="shadow-2xl"
                  loading={
                    <div className="flex items-center gap-3 text-white p-8">
                      <Loader2 size={24} className="animate-spin" />
                      <span>Loading page {currentPage}...</span>
                    </div>
                  }
                />
              )}
            </Document>
          )}
        </div>
      </div>
    </>
  );
}
