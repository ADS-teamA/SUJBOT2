/**
 * PDFSidePanel Component
 *
 * Right-side panel for viewing PDF documents with text selection.
 * Renders all pages as a scrollable document with lazy loading.
 *
 * Features:
 * - Seamless multi-page scrolling with lazy loading
 * - Text selection for agent context
 * - Zoom controls
 * - Keyboard navigation (Escape to close)
 * - Mobile fullscreen overlay mode
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import type { TextItem } from 'pdfjs-dist/types/src/display/api';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import {
  X,
  ZoomIn,
  ZoomOut,
  Loader2,
  AlertCircle,
  FileText,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import type { TextSelection } from '../../types';

// Configure PDF.js worker
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
pdfjs.GlobalWorkerOptions.workerSrc = pdfjsWorker;

// API base URL from environment
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// Minimum selection length to trigger context capture
const MIN_SELECTION_LENGTH = 10;

interface PDFSidePanelProps {
  isOpen: boolean;
  documentId: string;
  documentName: string;
  initialPage?: number;
  chunkContent?: string;
  onClose: () => void;
  onTextSelected: (selection: TextSelection) => void;
}

/**
 * Extract significant phrases from chunk content for fuzzy matching.
 */
function extractSearchPhrases(content: string, maxPhrases: number = 15): string[] {
  if (!content) return [];
  const normalized = content.slice(0, 800).toLowerCase().replace(/\s+/g, ' ').trim();
  const words = normalized.split(' ').filter(w => w.length > 2);
  if (words.length < 4) return [];
  const phrases: string[] = [];
  for (let i = 0; i < words.length - 3 && phrases.length < maxPhrases; i++) {
    phrases.push(words.slice(i, i + 4).join(' '));
  }
  return phrases;
}

function textContainsPhrase(text: string, phrases: string[]): boolean {
  if (!text || phrases.length === 0) return false;
  const normalizedText = text.toLowerCase().replace(/\s+/g, ' ');
  return phrases.some(phrase => normalizedText.includes(phrase));
}

export function PDFSidePanel({
  isOpen,
  documentId,
  documentName,
  initialPage = 1,
  chunkContent,
  onClose,
  onTextSelected,
}: PDFSidePanelProps) {
  const { t } = useTranslation();
  const [numPages, setNumPages] = useState<number | null>(null);
  const [scale, setScale] = useState(1.0);
  const [error, setError] = useState<string | null>(null);
  const [_isLoading, setIsLoading] = useState(true);
  const [pdfData, setPdfData] = useState<ArrayBuffer | null>(null);
  const [currentVisiblePage, setCurrentVisiblePage] = useState(initialPage);
  const [loadedPages, setLoadedPages] = useState<Set<number>>(new Set());

  const containerRef = useRef<HTMLDivElement>(null);
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Highlight phrases for chunk matching
  const highlightPhrases = useMemo(() => {
    if (!chunkContent) return [];
    return extractSearchPhrases(chunkContent);
  }, [chunkContent]);

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
      setLoadedPages(new Set());

      try {
        const response = await fetch(pdfUrl, {
          credentials: 'include',
        });

        if (!response.ok) {
          const errorMessages: Record<number, string> = {
            401: t('pdfPanel.error401', 'Session expired. Please log in again.'),
            403: t('pdfPanel.error403', 'You do not have permission to view this document.'),
            404: t('pdfPanel.error404', 'Document not found.'),
            500: t('pdfPanel.error500', 'Server temporarily unavailable.'),
          };
          const message = errorMessages[response.status] || t('pdfPanel.errorGeneric', `Error loading PDF (${response.status})`);
          throw new Error(message);
        }

        const arrayBuffer = await response.arrayBuffer();
        setPdfData(arrayBuffer);
      } catch (err) {
        const message = err instanceof Error ? err.message : t('pdfPanel.errorGeneric', 'Failed to load PDF');
        setError(message);
        setIsLoading(false);
      }
    };

    fetchPdf();
  }, [documentId, pdfUrl, isOpen, t]);

  // Reset state when document changes
  useEffect(() => {
    setError(null);
    setIsLoading(true);
    setNumPages(null);
    setPdfData(null);
    setCurrentVisiblePage(initialPage);
    setLoadedPages(new Set());
  }, [documentId, initialPage]);

  // Handle document load success
  const onDocumentLoadSuccess = useCallback(
    ({ numPages }: { numPages: number }) => {
      setNumPages(numPages);
      setIsLoading(false);
      setError(null);
      // Pre-load initial page and neighbors
      const pagesToLoad = new Set<number>();
      for (let i = Math.max(1, initialPage - 1); i <= Math.min(numPages, initialPage + 2); i++) {
        pagesToLoad.add(i);
      }
      setLoadedPages(pagesToLoad);
    },
    [initialPage]
  );

  // Handle document load error
  const onDocumentLoadError = useCallback((error: Error) => {
    setError(`Failed to load PDF: ${error.message}`);
    setIsLoading(false);
  }, []);

  // Setup IntersectionObserver for lazy loading pages
  useEffect(() => {
    if (!numPages || !containerRef.current) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          const pageNum = parseInt(entry.target.getAttribute('data-page') || '0');
          if (entry.isIntersecting && pageNum > 0) {
            // Load this page and neighbors
            setLoadedPages(prev => {
              const next = new Set(prev);
              for (let i = Math.max(1, pageNum - 1); i <= Math.min(numPages, pageNum + 2); i++) {
                next.add(i);
              }
              return next;
            });
            // Update current visible page (use the one with most visibility)
            if (entry.intersectionRatio > 0.5) {
              setCurrentVisiblePage(pageNum);
            }
          }
        });
      },
      {
        root: containerRef.current,
        rootMargin: '200px 0px',
        threshold: [0, 0.5, 1],
      }
    );

    // Observe all page placeholders
    pageRefs.current.forEach((ref, _pageNum) => {
      if (observerRef.current && ref) {
        observerRef.current.observe(ref);
      }
    });

    return () => {
      observerRef.current?.disconnect();
    };
  }, [numPages]);

  // Scroll to initial page after PDF loads
  useEffect(() => {
    if (!numPages || !containerRef.current || initialPage <= 1) return;

    const pageRef = pageRefs.current.get(initialPage);
    if (pageRef) {
      setTimeout(() => {
        pageRef.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    }
  }, [numPages, initialPage]);

  // Text selection handler
  const handleSelectionChange = useCallback(() => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed) return;

    const text = selection.toString().trim();
    if (text.length < MIN_SELECTION_LENGTH) return;

    // Check if selection is within this PDF panel
    const range = selection.getRangeAt(0);
    const container = containerRef.current;
    if (!container || !container.contains(range.commonAncestorContainer)) return;

    // Extract page numbers from selection
    const getPageFromNode = (node: Node): number => {
      let element = node.nodeType === Node.TEXT_NODE ? node.parentElement : node as Element;
      while (element && !element.getAttribute('data-page')) {
        element = element.parentElement;
      }
      return parseInt(element?.getAttribute('data-page') || '1');
    };

    const pageStart = getPageFromNode(range.startContainer);
    const pageEnd = getPageFromNode(range.endContainer);

    onTextSelected({
      text,
      documentId,
      documentName,
      pageStart,
      pageEnd,
      charCount: text.length,
    });
  }, [documentId, documentName, onTextSelected]);

  // Listen for selection changes
  useEffect(() => {
    if (!isOpen) return;
    document.addEventListener('mouseup', handleSelectionChange);
    return () => document.removeEventListener('mouseup', handleSelectionChange);
  }, [isOpen, handleSelectionChange]);

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          onClose();
          break;
        case '+':
        case '=':
          setScale(s => Math.min(3, s + 0.2));
          break;
        case '-':
          setScale(s => Math.max(0.5, s - 0.2));
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Custom text renderer for highlighting
  const customTextRenderer = useCallback(
    (textItem: TextItem) => {
      const str = textItem.str;
      if (!highlightPhrases.length || !str.trim()) return str;
      if (textContainsPhrase(str, highlightPhrases)) {
        return `<mark class="chunk-highlight">${str}</mark>`;
      }
      return str;
    },
    [highlightPhrases]
  );

  if (!isOpen) return null;

  // Mobile fullscreen overlay
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;

  return (
    <div
      className={cn(
        'flex flex-col',
        'bg-white dark:bg-accent-900',
        'border-l border-accent-200 dark:border-accent-700',
        // Mobile: fullscreen overlay
        isMobile && 'fixed inset-0 z-50',
        // Desktop: side panel (width controlled by parent)
        !isMobile && 'h-full'
      )}
    >
      {/* Header */}
      <div
        className={cn(
          'flex items-center justify-between',
          'px-4 py-3',
          'bg-accent-50 dark:bg-accent-800',
          'border-b border-accent-200 dark:border-accent-700',
          'flex-shrink-0'
        )}
      >
        {/* Document info */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <FileText size={20} className="text-accent-500 dark:text-accent-400 flex-shrink-0" />
          <h2 className="text-sm font-semibold text-accent-900 dark:text-accent-100 truncate">
            {documentName.replace(/_/g, ' ')}
          </h2>
        </div>

        {/* Page indicator */}
        {numPages && (
          <div className="text-xs text-accent-500 dark:text-accent-400 mx-4">
            {t('pdfPanel.page', 'Page')} {currentVisiblePage} / {numPages}
          </div>
        )}

        {/* Zoom controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setScale(s => Math.max(0.5, s - 0.2))}
            disabled={scale <= 0.5}
            className={cn(
              'p-1.5 rounded transition-colors',
              'hover:bg-accent-200 dark:hover:bg-accent-700',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
            title={t('pdfPanel.zoomOut', 'Zoom out (-)')}
          >
            <ZoomOut size={16} />
          </button>

          <span className="text-xs text-accent-600 dark:text-accent-400 w-12 text-center">
            {Math.round(scale * 100)}%
          </span>

          <button
            onClick={() => setScale(s => Math.min(3, s + 0.2))}
            disabled={scale >= 3}
            className={cn(
              'p-1.5 rounded transition-colors',
              'hover:bg-accent-200 dark:hover:bg-accent-700',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
            title={t('pdfPanel.zoomIn', 'Zoom in (+)')}
          >
            <ZoomIn size={16} />
          </button>
        </div>

        {/* Close button */}
        <button
          onClick={onClose}
          className={cn(
            'p-1.5 rounded transition-colors ml-2',
            'hover:bg-accent-200 dark:hover:bg-accent-700',
            'text-accent-500 dark:text-accent-400'
          )}
          title={t('pdfPanel.close', 'Close (Esc)')}
        >
          <X size={18} />
        </button>
      </div>

      {/* PDF content - scrollable */}
      <div
        ref={containerRef}
        className={cn(
          'flex-1 overflow-auto',
          'bg-accent-100 dark:bg-accent-950'
        )}
      >
        {error ? (
          <div className="flex flex-col items-center justify-center gap-4 p-8 text-red-500">
            <AlertCircle size={48} />
            <p className="text-lg font-medium text-center">{error}</p>
          </div>
        ) : !pdfData ? (
          <div className="flex items-center justify-center gap-3 p-8 text-accent-500">
            <Loader2 size={24} className="animate-spin" />
            <span>{t('pdfPanel.loading', 'Loading PDF...')}</span>
          </div>
        ) : (
          <Document
            file={pdfData}
            onLoadSuccess={onDocumentLoadSuccess}
            onLoadError={onDocumentLoadError}
            loading={
              <div className="flex items-center justify-center gap-3 p-8 text-accent-500">
                <Loader2 size={24} className="animate-spin" />
                <span>{t('pdfPanel.rendering', 'Rendering...')}</span>
              </div>
            }
            className="flex flex-col items-center py-4 gap-4"
          >
            {numPages && Array.from({ length: numPages }, (_, i) => i + 1).map(pageNum => (
              <div
                key={pageNum}
                ref={(el) => {
                  if (el) {
                    pageRefs.current.set(pageNum, el);
                    observerRef.current?.observe(el);
                  }
                }}
                data-page={pageNum}
                className="shadow-lg"
                style={{ minHeight: loadedPages.has(pageNum) ? undefined : '800px' }}
              >
                {loadedPages.has(pageNum) ? (
                  <Page
                    pageNumber={pageNum}
                    scale={scale}
                    renderTextLayer={true}
                    renderAnnotationLayer={true}
                    customTextRenderer={
                      pageNum === initialPage && highlightPhrases.length > 0
                        ? customTextRenderer
                        : undefined
                    }
                    loading={
                      <div className="flex items-center justify-center w-full h-96 bg-white">
                        <Loader2 size={24} className="animate-spin text-accent-400" />
                      </div>
                    }
                  />
                ) : (
                  <div className="flex items-center justify-center w-full h-[800px] bg-white dark:bg-accent-800">
                    <span className="text-accent-400 text-sm">
                      {t('pdfPanel.page', 'Page')} {pageNum}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </Document>
        )}
      </div>
    </div>
  );
}
