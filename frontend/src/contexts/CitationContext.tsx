/**
 * Citation Context Provider
 *
 * Manages citation metadata cache and PDF viewer modal state.
 * Provides batch fetching with debouncing for efficiency.
 */

import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import type { CitationMetadata, CitationContextValue } from '../types';
import { PDFViewerModal } from '../components/pdf/PDFViewerModal';

// API base URL from environment
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

const CitationContext = createContext<CitationContextValue | null>(null);

interface CitationProviderProps {
  children: React.ReactNode;
}

export function CitationProvider({ children }: CitationProviderProps) {
  // Citation metadata cache
  const [citationCache, setCitationCache] = useState<Map<string, CitationMetadata>>(new Map());

  // PDF viewer state
  const [activePdf, setActivePdf] = useState<{
    documentId: string;
    page: number;
    chunkId?: string;
  } | null>(null);

  // Loading state
  const [isLoading, setIsLoading] = useState(false);

  // Error state for user feedback
  const [error, setError] = useState<string | null>(null);

  // Batch fetch optimization
  const pendingFetches = useRef<Set<string>>(new Set());
  const fetchTimeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // Cleanup debounce timeout on unmount
  useEffect(() => {
    return () => {
      if (fetchTimeoutRef.current) {
        clearTimeout(fetchTimeoutRef.current);
      }
    };
  }, []);

  // Use ref to access cache without recreating callback
  const citationCacheRef = useRef(citationCache);
  useEffect(() => {
    citationCacheRef.current = citationCache;
  }, [citationCache]);

  /**
   * Fetch citation metadata for chunk IDs.
   * Uses batching with 100ms debounce for efficiency.
   * Uses refs to avoid recreating callback on cache updates.
   */
  const fetchCitationMetadata = useCallback(async (chunkIds: string[]) => {
    // Filter already cached (use ref to avoid dependency)
    const uncachedIds = chunkIds.filter(id => !citationCacheRef.current.has(id));
    if (uncachedIds.length === 0) return;

    // Add to pending set
    uncachedIds.forEach(id => pendingFetches.current.add(id));

    // Clear existing timeout
    if (fetchTimeoutRef.current) {
      clearTimeout(fetchTimeoutRef.current);
    }

    // Debounce batch fetch (wait 100ms for more IDs to accumulate)
    fetchTimeoutRef.current = setTimeout(async () => {
      const idsToFetch = Array.from(pendingFetches.current);
      pendingFetches.current.clear();

      if (idsToFetch.length === 0) return;

      setIsLoading(true);
      setError(null); // Clear previous errors

      try {
        const response = await fetch(`${API_BASE_URL}/api/citations/batch`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include', // Include cookies for auth
          body: JSON.stringify({ chunk_ids: idsToFetch }),
        });

        if (!response.ok) {
          const errorMsg = `Failed to fetch citations (${response.status})`;
          console.error('Failed to fetch citation metadata:', response.status);
          setError(errorMsg);
          return;
        }

        const data: Array<{
          chunk_id: string;
          document_id: string;
          document_name: string;
          section_title: string | null;
          section_path: string | null;
          hierarchical_path: string | null;
          page_number: number | null;
          pdf_available: boolean;
        }> = await response.json();

        // Update cache
        setCitationCache(prev => {
          const next = new Map(prev);
          data.forEach(item => {
            next.set(item.chunk_id, {
              chunkId: item.chunk_id,
              documentId: item.document_id,
              documentName: item.document_name,
              sectionTitle: item.section_title,
              sectionPath: item.section_path,
              hierarchicalPath: item.hierarchical_path,
              pageNumber: item.page_number,
              pdfAvailable: item.pdf_available,
            });
          });
          return next;
        });
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Network error';
        console.error('Error fetching citation metadata:', err);
        setError(`Citation fetch failed: ${errorMsg}`);
      } finally {
        setIsLoading(false);
      }
    }, 100); // Increased to 100ms for better batching
  }, []); // No dependencies - uses refs

  /**
   * Clear error state (for user dismissal).
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  /**
   * Open PDF viewer modal.
   */
  const openPdf = useCallback((documentId: string, page: number = 1, chunkId?: string) => {
    setActivePdf({ documentId, page, chunkId });
  }, []);

  /**
   * Close PDF viewer modal.
   */
  const closePdf = useCallback(() => {
    setActivePdf(null);
  }, []);

  const contextValue: CitationContextValue = {
    citationCache,
    activePdf,
    openPdf,
    closePdf,
    fetchCitationMetadata,
    isLoading,
    error,
    clearError,
  };

  return (
    <CitationContext.Provider value={contextValue}>
      {children}

      {/* PDF Viewer Modal - rendered at root level */}
      {activePdf && (
        <PDFViewerModal
          isOpen={true}
          documentId={activePdf.documentId}
          initialPage={activePdf.page}
          onClose={closePdf}
        />
      )}
    </CitationContext.Provider>
  );
}

/**
 * Hook to access citation context.
 * Must be used within CitationProvider.
 */
export function useCitationContext(): CitationContextValue {
  const context = useContext(CitationContext);
  if (!context) {
    throw new Error('useCitationContext must be used within CitationProvider');
  }
  return context;
}
