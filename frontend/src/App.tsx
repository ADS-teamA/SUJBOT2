/**
 * Main App Component with URL-based Routing
 *
 * Routes (based on window.location.pathname):
 * - / - Main chat application
 * - /admin/login - Admin login
 * - /admin/* - Admin portal (requires admin auth)
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { AlertTriangle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Header } from './components/header/Header';
import { Sidebar } from './components/sidebar/Sidebar';
import { ResponsiveSidebar } from './components/layout/ResponsiveSidebar';
import { ChatContainer } from './components/chat/ChatContainer';
import { PDFSidePanel } from './components/pdf/PDFSidePanel';
import { LoginPage } from './pages/LoginPage';
import { useChat } from './hooks/useChat';
import { useTheme } from './hooks/useTheme';
import { useAuth } from './contexts/AuthContext';
import { useCitationContext } from './contexts/CitationContext';
import { cn } from './design-system/utils/cn';
import { apiService } from './services/api';

// Admin imports
import { AdminLoginPage } from './admin/pages/AdminLoginPage';
import { AdminApp } from './admin/AdminApp';

import './index.css';

// Default PDF panel width (40%)
const DEFAULT_PDF_PANEL_WIDTH = 40;
const MIN_PDF_PANEL_WIDTH = 20;
const MAX_PDF_PANEL_WIDTH = 70;

/**
 * Main chat application component
 */
function MainApp() {
  const { t } = useTranslation();
  const { isAuthenticated, isLoading } = useAuth();
  const { activePdf, closePdf, citationCache, setSelectedText } = useCitationContext();

  const {
    conversations,
    currentConversation,
    isStreaming,
    clarificationData,
    awaitingClarification,
    spendingLimitError,
    spendingRefreshTrigger,
    createConversation,
    selectConversation,
    deleteConversation,
    renameConversation,
    sendMessage,
    editMessage,
    regenerateMessage,
    submitClarification,
    cancelClarification,
    cancelStreaming,
    clearSpendingLimitError,
  } = useChat();

  useTheme();

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [degradedComponents, setDegradedComponents] = useState<Array<{component: string; error: string}>>([]);

  // PDF panel width state (percentage)
  const [pdfPanelWidth, setPdfPanelWidth] = useState(() => {
    const saved = localStorage.getItem('pdfPanelWidth');
    return saved ? parseFloat(saved) : DEFAULT_PDF_PANEL_WIDTH;
  });
  const [isResizing, setIsResizing] = useState(false);
  const [isPdfClosing, setIsPdfClosing] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Handle PDF panel close with animation
  const handleClosePdf = useCallback(() => {
    setIsPdfClosing(true);
    // Wait for animation to complete before actually closing
    setTimeout(() => {
      closePdf();
      setIsPdfClosing(false);
    }, 300); // Match animation duration
  }, [closePdf]);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

  // Save panel width to localStorage
  useEffect(() => {
    localStorage.setItem('pdfPanelWidth', String(pdfPanelWidth));
  }, [pdfPanelWidth]);

  // Handle resize drag
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return;
      const containerRect = containerRef.current.getBoundingClientRect();
      const newWidth = ((containerRect.right - e.clientX) / containerRect.width) * 100;
      setPdfPanelWidth(Math.max(MIN_PDF_PANEL_WIDTH, Math.min(MAX_PDF_PANEL_WIDTH, newWidth)));
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  useEffect(() => {
    apiService.checkHealth()
      .then((health) => {
        if (health.status === 'degraded' && health.degraded_components) {
          setDegradedComponents(health.degraded_components);
        }
      })
      .catch((error) => {
        console.error('Health check failed:', error);
      });
  }, []);

  if (isLoading) {
    return (
      <div className={cn(
        'h-screen flex items-center justify-center',
        'bg-white dark:bg-accent-950'
      )}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent-900 dark:border-accent-100 mx-auto mb-4"></div>
          <p className="text-accent-600 dark:text-accent-400">{t('common.verifyingSession')}</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return (
    <div className={cn(
      'h-screen flex flex-col',
      'bg-white dark:bg-accent-950',
      'text-accent-900 dark:text-accent-100'
    )}>
      <Header
        onToggleSidebar={toggleSidebar}
        sidebarOpen={sidebarOpen}
      />

      {degradedComponents.length > 0 && (
        <div className={cn(
          'px-4 py-3 border-b',
          'bg-yellow-50 dark:bg-yellow-900/20',
          'border-yellow-200 dark:border-yellow-800',
          'text-yellow-800 dark:text-yellow-200'
        )}>
          <div className="flex items-start gap-3 max-w-7xl mx-auto">
            <AlertTriangle size={20} className="flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <div className="font-semibold text-sm mb-1">
                {t('degradedMode.title')}
              </div>
              <div className="text-xs opacity-90">
                {t('degradedMode.description', { components: degradedComponents.map(d => d.component).join(', ') })}
              </div>
            </div>
          </div>
        </div>
      )}

      <div ref={containerRef} className="flex-1 flex overflow-hidden">
        <ResponsiveSidebar isOpen={sidebarOpen} onToggle={toggleSidebar}>
          <Sidebar
            conversations={conversations}
            currentConversationId={currentConversation?.id || null}
            onSelectConversation={selectConversation}
            onNewConversation={createConversation}
            onDeleteConversation={deleteConversation}
            onRenameConversation={renameConversation}
          />
        </ResponsiveSidebar>

        {/* Chat area - flex grows, shrinks when PDF panel is open */}
        <div className="flex-1 min-w-0 flex flex-col">
          <ChatContainer
            conversation={currentConversation}
            isStreaming={isStreaming}
            onSendMessage={sendMessage}
            onEditMessage={editMessage}
            onRegenerateMessage={regenerateMessage}
            onCancelStreaming={cancelStreaming}
            clarificationData={clarificationData}
            awaitingClarification={awaitingClarification}
            onSubmitClarification={submitClarification}
            onCancelClarification={cancelClarification}
            spendingRefreshTrigger={spendingRefreshTrigger}
            spendingLimitError={spendingLimitError}
            onClearSpendingLimitError={clearSpendingLimitError}
          />
        </div>

        {/* PDF Side Panel with resize handle and close animation */}
        {(activePdf || isPdfClosing) && (
          <>
            {/* Resize handle - hide during closing animation */}
            {!isPdfClosing && (
              <div
                onMouseDown={handleResizeStart}
                className={cn(
                  'w-1.5 cursor-col-resize flex-shrink-0',
                  'bg-accent-200 dark:bg-accent-700',
                  'hover:bg-blue-400 dark:hover:bg-blue-600',
                  'transition-colors duration-150',
                  isResizing && 'bg-blue-500 dark:bg-blue-500'
                )}
              />
            )}

            {/* PDF Panel with slide + fade animation */}
            <div
              style={{ width: `${pdfPanelWidth}%` }}
              className={cn(
                'flex-shrink-0 hidden md:block',
                'transition-all duration-300 ease-out',
                isPdfClosing && 'opacity-0 translate-x-8'
              )}
            >
              <PDFSidePanel
                isOpen={!isPdfClosing}
                documentId={activePdf?.documentId ?? ''}
                documentName={activePdf?.documentName ?? ''}
                initialPage={activePdf?.page ?? 1}
                chunkContent={activePdf?.chunkId ? citationCache.get(activePdf.chunkId)?.content ?? undefined : undefined}
                onClose={handleClosePdf}
                onTextSelected={setSelectedText}
              />
            </div>

            {/* Mobile fullscreen overlay with fade animation */}
            <div className={cn(
              'md:hidden fixed inset-0 z-50',
              'transition-opacity duration-300 ease-out',
              isPdfClosing && 'opacity-0'
            )}>
              <PDFSidePanel
                isOpen={!isPdfClosing}
                documentId={activePdf?.documentId ?? ''}
                documentName={activePdf?.documentName ?? ''}
                initialPage={activePdf?.page ?? 1}
                chunkContent={activePdf?.chunkId ? citationCache.get(activePdf.chunkId)?.content ?? undefined : undefined}
                onClose={handleClosePdf}
                onTextSelected={setSelectedText}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/**
 * Admin guard component - redirects to admin login if not admin
 */
function AdminGuard({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, user } = useAuth();

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-accent-50 dark:bg-accent-950">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent-900 dark:border-accent-100"></div>
      </div>
    );
  }

  if (!isAuthenticated || !user?.is_admin) {
    window.location.href = '/admin/login';
    return null;
  }

  return <>{children}</>;
}

/**
 * App with URL-based routing (no react-router-dom dependency)
 */
function App() {
  const pathname = window.location.pathname;

  // Admin login page
  if (pathname === '/admin/login') {
    return <AdminLoginPage />;
  }

  // Admin portal (protected)
  if (pathname.startsWith('/admin')) {
    return (
      <AdminGuard>
        <AdminApp />
      </AdminGuard>
    );
  }

  // Main application (default)
  return <MainApp />;
}

export default App;
