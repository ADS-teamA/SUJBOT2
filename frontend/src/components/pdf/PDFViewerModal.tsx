/**
 * PDFViewerModal Component
 *
 * Fullscreen modal overlay for viewing PDF documents.
 * Uses react-pdf for rendering with page navigation.
 *
 * Features:
 * - Single page view with navigation arrows
 * - Zoom controls
 * - Keyboard navigation (arrows, +/-, Escape)
 * - Page number input
 * - Auto-scroll to initial page from citation context
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import {
  X,
  ZoomIn,
  ZoomOut,
  ChevronLeft,
  ChevronRight,
  Loader2,
  AlertCircle,
  FileText,
} from 'lucide-react';
import { cn } from '../../design-system/utils/cn';

// Configure PDF.js worker
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
pdfjs.GlobalWorkerOptions.workerSrc = pdfjsWorker;

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
  const [pdfData, setPdfData] = useState<{ data: ArrayBuffer } | null>(null);
  const [pageInputValue, setPageInputValue] = useState(String(initialPage));

  const containerRef = useRef<HTMLDivElement>(null);

  // PDF URL
  const pdfUrl = documentId
    ? `${API_BASE_URL}/api/documents/${encodeURIComponent(documentId)}/pdf`
    : '';

  // Fetch PDF with credentials
  useEffect(() => {
    if (!documentId || !isOpen) return;

    const fetchPdf = async () => {
      setIsLoading(true);
      setError(null);
      setPdfData(null);

      try {
        console.log('[PDFViewerModal] Fetching PDF:', pdfUrl);
        const response = await fetch(pdfUrl, {
          credentials: 'include',
        });

        if (!response.ok) {
          // Map HTTP status to user-friendly message
          const errorMessages: Record<number, string> = {
            401: 'Vaše relace vypršela. Přihlaste se prosím znovu.',
            403: 'Nemáte oprávnění zobrazit tento dokument.',
            404: 'Dokument nebyl nalezen nebo již není dostupný.',
            500: 'Server je dočasně nedostupný. Zkuste to prosím později.',
            502: 'Server je dočasně nedostupný. Zkuste to prosím později.',
            503: 'Server je přetížen. Zkuste to prosím později.',
          };
          const message = errorMessages[response.status] || `Chyba při načítání PDF (${response.status})`;
          throw new Error(message);
        }

        const arrayBuffer = await response.arrayBuffer();
        console.log('[PDFViewerModal] PDF fetched, size:', arrayBuffer.byteLength);
        setPdfData({ data: arrayBuffer });
      } catch (err) {
        console.error('[PDFViewerModal] PDF fetch error:', err);
        const message = err instanceof Error ? err.message : 'Nepodařilo se načíst PDF';
        setError(message);
        setIsLoading(false);
      }
    };

    fetchPdf();
  }, [documentId, pdfUrl, isOpen]);

  // Reset state when document changes
  useEffect(() => {
    setError(null);
    setIsLoading(true);
    setNumPages(null);
    setPdfData(null);
    setCurrentPage(initialPage);
    setPageInputValue(String(initialPage));
  }, [documentId, initialPage]);

  // Handle document load success
  const onDocumentLoadSuccess = useCallback(
    ({ numPages }: { numPages: number }) => {
      setNumPages(numPages);
      setIsLoading(false);
      setError(null);
      // Ensure current page is within bounds
      if (initialPage > numPages) {
        setCurrentPage(numPages);
        setPageInputValue(String(numPages));
      }
    },
    [initialPage]
  );

  // Handle document load error
  const onDocumentLoadError = useCallback((error: Error) => {
    console.error('[PDFViewerModal] PDF load error:', error.message, error);
    setError(`Failed to load PDF: ${error.message}`);
    setIsLoading(false);
  }, []);

  // Navigation
  const goToPage = useCallback(
    (page: number) => {
      if (numPages && page >= 1 && page <= numPages) {
        setCurrentPage(page);
        setPageInputValue(String(page));
      }
    },
    [numPages]
  );

  const prevPage = useCallback(() => {
    goToPage(currentPage - 1);
  }, [currentPage, goToPage]);

  const nextPage = useCallback(() => {
    goToPage(currentPage + 1);
  }, [currentPage, goToPage]);

  // Zoom handlers
  const zoomIn = useCallback(() => {
    setScale((s) => Math.min(3, s + 0.2));
  }, []);

  const zoomOut = useCallback(() => {
    setScale((s) => Math.max(0.5, s - 0.2));
  }, []);

  // Page input handler
  const handlePageInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPageInputValue(e.target.value);
  };

  const handlePageInputBlur = () => {
    const page = parseInt(pageInputValue, 10);
    if (!isNaN(page) && numPages) {
      goToPage(Math.max(1, Math.min(numPages, page)));
    } else {
      setPageInputValue(String(currentPage));
    }
  };

  const handlePageInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handlePageInputBlur();
    }
  };

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          onClose();
          break;
        case 'ArrowLeft':
          prevPage();
          break;
        case 'ArrowRight':
          nextPage();
          break;
        case 'Home':
          goToPage(1);
          break;
        case 'End':
          if (numPages) goToPage(numPages);
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
  }, [isOpen, onClose, prevPage, nextPage, goToPage, numPages, zoomIn, zoomOut]);

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
        className={cn('fixed inset-0 z-50', 'flex flex-col', 'pointer-events-none')}
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
            <FileText size={20} className="text-accent-500 dark:text-accent-400" />
            <h2 className="text-lg font-semibold text-accent-900 dark:text-accent-100">
              {documentId.replace(/_/g, ' ')}
            </h2>
          </div>

          {/* Page navigation */}
          <div className="flex items-center gap-2">
            <button
              onClick={prevPage}
              disabled={currentPage <= 1}
              className={cn(
                'p-2 rounded-lg transition-colors',
                'hover:bg-accent-100 dark:hover:bg-accent-800',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
              title="Previous page (←)"
            >
              <ChevronLeft size={20} />
            </button>

            <div className="flex items-center gap-1 text-sm">
              <input
                type="text"
                value={pageInputValue}
                onChange={handlePageInputChange}
                onBlur={handlePageInputBlur}
                onKeyDown={handlePageInputKeyDown}
                className={cn(
                  'w-12 px-2 py-1 text-center rounded',
                  'bg-accent-100 dark:bg-accent-800',
                  'border border-accent-300 dark:border-accent-600',
                  'text-accent-900 dark:text-accent-100',
                  'focus:outline-none focus:ring-2 focus:ring-blue-500'
                )}
              />
              <span className="text-accent-500 dark:text-accent-400">
                / {numPages || '...'}
              </span>
            </div>

            <button
              onClick={nextPage}
              disabled={!numPages || currentPage >= numPages}
              className={cn(
                'p-2 rounded-lg transition-colors',
                'hover:bg-accent-100 dark:hover:bg-accent-800',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
              title="Next page (→)"
            >
              <ChevronRight size={20} />
            </button>
          </div>

          {/* Zoom controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={zoomOut}
              disabled={scale <= 0.5}
              className={cn(
                'p-2 rounded-lg transition-colors',
                'hover:bg-accent-100 dark:hover:bg-accent-800',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
              title="Zoom out (-)"
            >
              <ZoomOut size={20} />
            </button>

            <span className="text-sm text-accent-600 dark:text-accent-400 min-w-[4rem] text-center">
              {Math.round(scale * 100)}%
            </span>

            <button
              onClick={zoomIn}
              disabled={scale >= 3}
              className={cn(
                'p-2 rounded-lg transition-colors',
                'hover:bg-accent-100 dark:hover:bg-accent-800',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
              title="Zoom in (+)"
            >
              <ZoomIn size={20} />
            </button>
          </div>

          {/* Close button */}
          <button
            onClick={onClose}
            className={cn(
              'p-2 rounded-lg transition-colors',
              'hover:bg-accent-100 dark:hover:bg-accent-800',
              'text-accent-500 dark:text-accent-400'
            )}
            title="Close (Esc)"
          >
            <X size={20} />
          </button>
        </div>

        {/* PDF content */}
        <div
          ref={containerRef}
          className={cn(
            'flex-1 overflow-auto',
            'bg-accent-800 dark:bg-accent-950',
            'pointer-events-auto'
          )}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex justify-center py-4 min-h-full">
            {error ? (
              <div
                className={cn(
                  'flex flex-col items-center justify-center gap-4',
                  'p-8',
                  'text-red-400'
                )}
              >
                <AlertCircle size={48} />
                <p className="text-lg font-medium">{error}</p>
                <p className="text-sm text-accent-400">Document: {documentId}</p>
              </div>
            ) : !pdfData ? (
              <div className="flex items-center justify-center gap-3 text-white">
                <Loader2 size={24} className="animate-spin" />
                <span>Loading PDF...</span>
              </div>
            ) : (
              <Document
                file={pdfData}
                onLoadSuccess={onDocumentLoadSuccess}
                onLoadError={onDocumentLoadError}
                loading={
                  <div className="flex items-center justify-center gap-3 text-white">
                    <Loader2 size={24} className="animate-spin" />
                    <span>Rendering...</span>
                  </div>
                }
              >
                <Page
                  pageNumber={currentPage}
                  scale={scale}
                  renderTextLayer={true}
                  renderAnnotationLayer={true}
                  className="shadow-2xl"
                  loading={
                    <div className="flex items-center justify-center w-full h-96 text-white">
                      <Loader2 size={24} className="animate-spin" />
                    </div>
                  }
                />
              </Document>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
