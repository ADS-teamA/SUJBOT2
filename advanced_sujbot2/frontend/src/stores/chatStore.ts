import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { ChatMessage } from '@/types';
import { chatWebSocket } from '@/services/websocket';
import { generateId } from '@/utils/file';
import { useDocumentStore } from './documentStore';

interface ChatStore {
  messages: ChatMessage[];
  isTyping: boolean;
  currentStreamingMessageId: string | null;

  sendMessage: (text: string) => Promise<void>;
  addMessage: (message: ChatMessage) => void;
  updateStreamingMessage: (chunk: string) => void;
  finishStreaming: () => void;
  clearChat: () => void;
}

export const useChatStore = create<ChatStore>()(
  devtools((set, get) => ({
    messages: [],
    isTyping: false,
    currentStreamingMessageId: null,

    sendMessage: async (text) => {
      const userMessage: ChatMessage = {
        id: generateId(),
        type: 'user',
        content: text,
        timestamp: new Date()
      };

      set((state) => ({
        messages: [...state.messages, userMessage]
      }));

      const botMessageId = generateId();
      const botMessage: ChatMessage = {
        id: botMessageId,
        type: 'bot',
        content: '',
        timestamp: new Date(),
        isStreaming: true
      };

      set((state) => ({
        messages: [...state.messages, botMessage],
        isTyping: true,
        currentStreamingMessageId: botMessageId
      }));

      try {
        // Get indexed documents from document store
        const documentStore = useDocumentStore.getState();
        const indexedDocuments = [
          ...documentStore.contracts.filter(d => d.status === 'indexed'),
          ...documentStore.laws.filter(d => d.status === 'indexed')
        ].map(d => d.id);

        // Send message with document context
        chatWebSocket.sendMessage(text, indexedDocuments);
      } catch (error: any) {
        set((state) => ({
          messages: state.messages.map(m =>
            m.id === botMessageId
              ? { ...m, content: 'Error: ' + error.message, isStreaming: false }
              : m
          ),
          isTyping: false,
          currentStreamingMessageId: null
        }));
      }
    },

    updateStreamingMessage: (chunk) => {
      const { currentStreamingMessageId } = get();
      if (!currentStreamingMessageId) return;

      set((state) => ({
        messages: state.messages.map(m =>
          m.id === currentStreamingMessageId
            ? { ...m, content: m.content + chunk }
            : m
        )
      }));
    },

    finishStreaming: () => {
      const { currentStreamingMessageId } = get();
      if (!currentStreamingMessageId) return;

      set((state) => ({
        messages: state.messages.map(m =>
          m.id === currentStreamingMessageId
            ? { ...m, isStreaming: false }
            : m
        ),
        isTyping: false,
        currentStreamingMessageId: null
      }));
    },

    addMessage: (message) => {
      set((state) => ({
        messages: [...state.messages, message]
      }));
    },

    clearChat: () => {
      set({ messages: [] });
    }
  }))
);
