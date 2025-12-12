/**
 * DocumentBrowser Component
 *
 * Dropdown menu for browsing and selecting available PDF documents.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { FileText, FolderOpen, X, Loader2, RefreshCw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { apiService } from '../../services/api';
import type { DocumentInfo } from '../../types';
import { useCitationContext } from '../../contexts/CitationContext';

interface DocumentBrowserProps {
  isOpen: boolean;
  onClose: () => void;
}

export function DocumentBrowser({ isOpen, onClose }: DocumentBrowserProps) {
  const { t } = useTranslation();
  const { openPdf } = useCitationContext();
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Load documents when opening
  useEffect(() => {
    if (isOpen) {
      loadDocuments();
    }
  }, [isOpen]);

  // Handle click outside to close
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    // Delay to prevent immediate close on open click
    const timeoutId = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside);
    }, 100);

    return () => {
      clearTimeout(timeoutId);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, onClose]);

  const loadDocuments = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const docs = await apiService.getDocuments();
      setDocuments(docs);
    } catch (err) {
      console.error('Failed to load documents:', err);
      setError(t('documentBrowser.loadError'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectDocument = useCallback((doc: DocumentInfo) => {
    openPdf(doc.document_id, doc.display_name, 1);
    onClose();
  }, [openPdf, onClose]);

  const formatSize = (bytes: number): string => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(0)} KB`;
    }
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  if (!isOpen) return null;

  return (
    <div
      ref={dropdownRef}
      className={cn(
        'absolute top-full right-0 mt-2',
        'w-80 max-h-96 overflow-hidden',
        'bg-white dark:bg-accent-900',
        'border border-accent-200 dark:border-accent-700',
        'rounded-xl shadow-lg',
        'z-50',
        'animate-in fade-in slide-in-from-top-2 duration-200'
      )}
    >
      {/* Header */}
      <div className={cn(
        'flex items-center justify-between',
        'px-4 py-3',
        'border-b border-accent-200 dark:border-accent-700',
        'bg-accent-50 dark:bg-accent-800/50'
      )}>
        <div className="flex items-center gap-2">
          <FolderOpen size={18} className="text-accent-600 dark:text-accent-400" />
          <h3 className="font-medium text-sm text-accent-900 dark:text-accent-100">
            {t('documentBrowser.title')}
          </h3>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={loadDocuments}
            disabled={isLoading}
            className={cn(
              'p-1.5 rounded-lg',
              'text-accent-500 hover:text-accent-700',
              'dark:text-accent-400 dark:hover:text-accent-200',
              'hover:bg-accent-100 dark:hover:bg-accent-700',
              'transition-colors',
              'disabled:opacity-50'
            )}
            title="Refresh"
          >
            <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={onClose}
            className={cn(
              'p-1.5 rounded-lg',
              'text-accent-500 hover:text-accent-700',
              'dark:text-accent-400 dark:hover:text-accent-200',
              'hover:bg-accent-100 dark:hover:bg-accent-700',
              'transition-colors'
            )}
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="overflow-y-auto max-h-72">
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="animate-spin text-accent-400" size={24} />
          </div>
        )}

        {error && !isLoading && (
          <div className="px-4 py-6 text-center">
            <p className="text-red-500 dark:text-red-400 text-sm mb-2">{error}</p>
            <button
              onClick={loadDocuments}
              className={cn(
                'text-xs px-3 py-1.5 rounded-lg',
                'bg-accent-100 dark:bg-accent-800',
                'hover:bg-accent-200 dark:hover:bg-accent-700',
                'text-accent-700 dark:text-accent-300',
                'transition-colors'
              )}
            >
              {t('common.retry') || 'Retry'}
            </button>
          </div>
        )}

        {!isLoading && !error && documents.length === 0 && (
          <div className="px-4 py-6 text-center text-accent-500 dark:text-accent-400 text-sm">
            {t('documentBrowser.noDocuments')}
          </div>
        )}

        {!isLoading && !error && documents.length > 0 && (
          <div className="p-2">
            {documents.map((doc) => (
              <button
                key={doc.document_id}
                onClick={() => handleSelectDocument(doc)}
                className={cn(
                  'w-full flex items-center gap-3 px-3 py-2.5',
                  'hover:bg-accent-100 dark:hover:bg-accent-800',
                  'rounded-lg transition-colors',
                  'text-left group'
                )}
              >
                <FileText
                  size={18}
                  className="text-accent-400 group-hover:text-accent-600 dark:group-hover:text-accent-300 transition-colors flex-shrink-0"
                />
                <div className="flex-1 min-w-0">
                  <div className="truncate font-medium text-sm text-accent-800 dark:text-accent-200">
                    {doc.display_name}
                  </div>
                  <div className="text-xs text-accent-500 dark:text-accent-400">
                    {formatSize(doc.size_bytes)}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
