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
  const { updateStreamingMessage, updatePipelineStatus, attachSources, finishStreaming } = useChatStore();
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
      } else if (data.type === 'pipeline_status') {
        // Update pipeline status
        updatePipelineStatus({
          pipeline: data.pipeline,
          stage: data.stage,
          stage_name: data.stage_name,
          step: data.step,
          total_steps: data.total_steps,
          progress: data.progress,
          message: data.message
        });
      } else if (data.type === 'sources') {
        // Attach sources to current message
        if (data.sources && Array.isArray(data.sources)) {
          attachSources(data.sources);
        }
      } else if (data.type === 'stream_complete') {
        finishStreaming();
      }
    };

    chatWebSocket.onMessage(handleMessage);

    return () => {
      chatWebSocket.removeMessageHandler(handleMessage);
      chatWebSocket.disconnect();
    };
  }, [updateStreamingMessage, updatePipelineStatus, attachSources, finishStreaming]);

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
