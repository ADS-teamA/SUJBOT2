/**
 * Main App Component
 *
 * Wires together:
 * - Header (with model selector and sidebar toggle)
 * - ResponsiveSidebar (conversation history with collapsible behavior)
 * - ChatContainer (messages and input)
 *
 * Uses custom hooks:
 * - useChat: Manages conversation state and SSE streaming
 * - useTheme: Applies light theme
 */

import { useState, useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
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
import './index.css';

function App() {
  // Authentication
  const { isAuthenticated, isLoading } = useAuth();

  // Custom hooks
  const {
    conversations,
    currentConversation,
    isStreaming,
    clarificationData,
    awaitingClarification,
    createConversation,
    selectConversation,
    deleteConversation,
    sendMessage,
    editMessage,
    regenerateMessage,
    submitClarification,
    cancelClarification,
  } = useChat();

  useTheme(); // Apply light theme

  // Sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Health status state
  const [degradedComponents, setDegradedComponents] = useState<Array<{component: string; error: string}>>([]);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

  // Check health status on mount
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

  // Show loading state while verifying token
  if (isLoading) {
    return (
      <div className={cn(
        'h-screen flex items-center justify-center',
        'bg-white dark:bg-accent-950'
      )}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent-900 dark:border-accent-100 mx-auto mb-4"></div>
          <p className="text-accent-600 dark:text-accent-400">Verifying session...</p>
        </div>
      </div>
    );
  }

  // Show login page if not authenticated
  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return (
    <div className={cn(
      'h-screen flex flex-col',
      'bg-white dark:bg-accent-950',
      'text-accent-900 dark:text-accent-100'
    )}>
      {/* Header */}
      <Header
        onToggleSidebar={toggleSidebar}
        sidebarOpen={sidebarOpen}
      />

      {/* Degraded Mode Warning Banner */}
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
                Running in Degraded Mode
              </div>
              <div className="text-xs opacity-90">
                Some features are unavailable: {degradedComponents.map(d => d.component).join(', ')}.
                {' '}Search quality may be reduced without reranking. Knowledge graph features are disabled.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Responsive Sidebar */}
        <ResponsiveSidebar isOpen={sidebarOpen} onToggle={toggleSidebar}>
          <Sidebar
            conversations={conversations}
            currentConversationId={currentConversation?.id || null}
            onSelectConversation={selectConversation}
            onNewConversation={createConversation}
            onDeleteConversation={deleteConversation}
          />
        </ResponsiveSidebar>

        {/* Chat area */}
        <ChatContainer
          conversation={currentConversation}
          isStreaming={isStreaming}
          onSendMessage={sendMessage}
          onEditMessage={editMessage}
          onRegenerateMessage={regenerateMessage}
          clarificationData={clarificationData}
          awaitingClarification={awaitingClarification}
          onSubmitClarification={submitClarification}
          onCancelClarification={cancelClarification}
        />
      </div>
    </div>
  );
}

export default App;
