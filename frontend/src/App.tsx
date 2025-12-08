/**
 * Main App Component with URL-based Routing
 *
 * Routes (based on window.location.pathname):
 * - / - Main chat application
 * - /admin/login - Admin login
 * - /admin/* - Admin portal (requires admin auth)
 */

import { useState, useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Header } from './components/header/Header';
import { Sidebar } from './components/sidebar/Sidebar';
import { ResponsiveSidebar } from './components/layout/ResponsiveSidebar';
import { ChatContainer } from './components/chat/ChatContainer';
import { LoginPage } from './pages/LoginPage';
import { useChat } from './hooks/useChat';
import { useTheme } from './hooks/useTheme';
import { useAuth } from './contexts/AuthContext';
import { cn } from './design-system/utils/cn';
import { apiService } from './services/api';

// Admin imports
import { AdminLoginPage } from './admin/pages/AdminLoginPage';
import { AdminApp } from './admin/AdminApp';

import './index.css';

/**
 * Main chat application component
 */
function MainApp() {
  const { t } = useTranslation();
  const { isAuthenticated, isLoading } = useAuth();

  const {
    conversations,
    currentConversation,
    isStreaming,
    clarificationData,
    awaitingClarification,
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
  } = useChat();

  useTheme();

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [degradedComponents, setDegradedComponents] = useState<Array<{component: string; error: string}>>([]);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

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

      <div className="flex-1 flex overflow-hidden">
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
        />
      </div>
    </div>
  );
}

/**
 * Admin guard component - redirects to admin login if not admin
 */
function AdminGuard({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, user } = useAuth();

  // Debug logging
  console.log('AdminGuard state:', { isLoading, isAuthenticated, user, is_admin: user?.is_admin });

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-accent-50 dark:bg-accent-950">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent-900 dark:border-accent-100"></div>
      </div>
    );
  }

  if (!isAuthenticated || !user?.is_admin) {
    console.log('AdminGuard: Redirecting to /admin/login - not authenticated or not admin');
    // Redirect to admin login
    window.location.href = '/admin/login';
    return null;
  }

  console.log('AdminGuard: Access granted, rendering children');
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
