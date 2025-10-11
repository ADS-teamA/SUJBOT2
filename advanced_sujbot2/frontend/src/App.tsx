import React, { useEffect } from 'react';
import { TopBar } from './components/TopBar';
import { DocumentPanel } from './components/DocumentPanel';
import { ChatArea } from './components/ChatArea';
import { Toaster } from './components/Toaster';
import { chatWebSocket } from './services/websocket';
import { useChatStore } from './stores/chatStore';
import { useUIStore } from './stores/uiStore';
import { useDocumentStore } from './stores/documentStore';

function App() {
  const { updateStreamingMessage, finishStreaming } = useChatStore();
  const { theme } = useUIStore();
  const { resumePolling } = useDocumentStore();

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  // Resume polling for documents on mount
  useEffect(() => {
    resumePolling();
  }, [resumePolling]);

  useEffect(() => {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
    chatWebSocket.connect(wsUrl);

    const handleMessage = (data: any) => {
      if (data.type === 'stream_chunk') {
        updateStreamingMessage(data.content);
      } else if (data.type === 'stream_complete') {
        finishStreaming();
      } else if (data.type === 'sources') {
        // Handle sources (already included in stream)
        console.log('Received sources:', data.sources);
      }
    };

    chatWebSocket.onMessage(handleMessage);

    return () => {
      chatWebSocket.removeMessageHandler(handleMessage);
      chatWebSocket.disconnect();
    };
  }, [updateStreamingMessage, finishStreaming]);

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-gray-50 via-white to-gray-50 dark:from-gray-950 dark:via-black dark:to-gray-950 transition-all duration-500">
      <TopBar />

      <div className="flex-1 grid grid-cols-[300px_1fr_300px] overflow-hidden gap-px bg-gradient-to-r from-gray-200 via-gray-300 to-gray-200 dark:from-gray-800 dark:via-gray-700 dark:to-gray-800">
        <DocumentPanel type="contract" />
        <ChatArea />
        <DocumentPanel type="law" />
      </div>

      <Toaster />
    </div>
  );
}

export default App;
