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
  editingMessageId: string | null; // NEW: Track which message is being edited

  sendMessage: (text: string, documentIds?: string[]) => Promise<void>;
  addMessage: (message: ChatMessage) => void;
  updateStreamingMessage: (chunk: string) => void;
  finishStreaming: () => void;
  clearChat: () => void;

  // NEW: Edit and regenerate functionality
  editMessage: (messageId: string, newText: string) => Promise<void>;
  regenerateResponse: (messageId: string) => Promise<void>;
  setEditingMessage: (messageId: string | null) => void;
  deleteMessage: (messageId: string) => void;
}

export const useChatStore = create<ChatStore>()(
  devtools((set, get) => ({
    messages: [],
    isTyping: false,
    currentStreamingMessageId: null,
    editingMessageId: null,

    sendMessage: async (text, documentIds) => {
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
        // Use provided documentIds or get selected documents from store
        const documentStore = useDocumentStore.getState();
        const docsToUse = documentIds || documentStore.selectedDocumentIds;

        // If no documents selected, use all indexed documents
        const finalDocumentIds = docsToUse.length > 0
          ? docsToUse
          : [
              ...documentStore.contracts.filter(d => d.status === 'indexed'),
              ...documentStore.laws.filter(d => d.status === 'indexed')
            ].map(d => d.id);

        // Send message with document context
        chatWebSocket.sendMessage(text, finalDocumentIds);
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
    },

    // NEW: Edit message functionality
    editMessage: async (messageId, newText) => {
      const { messages } = get();
      const messageIndex = messages.findIndex(m => m.id === messageId);

      if (messageIndex === -1) return;

      // Remove all messages after the edited message (including bot response)
      const messagesToKeep = messages.slice(0, messageIndex);
      set({ messages: messagesToKeep, editingMessageId: null });

      // Resend the edited message
      await get().sendMessage(newText);
    },

    // NEW: Regenerate bot response
    regenerateResponse: async (botMessageId) => {
      const { messages } = get();
      const botMessageIndex = messages.findIndex(m => m.id === botMessageId);

      if (botMessageIndex === -1 || botMessageIndex === 0) return;

      // Find the user message that triggered this bot response
      let userMessageIndex = botMessageIndex - 1;
      while (userMessageIndex >= 0 && messages[userMessageIndex].type !== 'user') {
        userMessageIndex--;
      }

      if (userMessageIndex < 0) return;

      const userMessage = messages[userMessageIndex];

      // Remove the bot message
      set((state) => ({
        messages: state.messages.filter(m => m.id !== botMessageId)
      }));

      // Create a new bot message for the regenerated response
      const newBotMessageId = generateId();
      const newBotMessage: ChatMessage = {
        id: newBotMessageId,
        type: 'bot',
        content: '',
        timestamp: new Date(),
        isStreaming: true
      };

      set((state) => ({
        messages: [...state.messages, newBotMessage],
        isTyping: true,
        currentStreamingMessageId: newBotMessageId
      }));

      try {
        // Get selected documents or use all indexed
        const documentStore = useDocumentStore.getState();
        const docsToUse = documentStore.selectedDocumentIds;

        const finalDocumentIds = docsToUse.length > 0
          ? docsToUse
          : [
              ...documentStore.contracts.filter(d => d.status === 'indexed'),
              ...documentStore.laws.filter(d => d.status === 'indexed')
            ].map(d => d.id);

        // Send the same query again WITHOUT adding a new user message
        chatWebSocket.sendMessage(userMessage.content, finalDocumentIds);
      } catch (error: any) {
        set((state) => ({
          messages: state.messages.map(m =>
            m.id === newBotMessageId
              ? { ...m, content: 'Error: ' + error.message, isStreaming: false }
              : m
          ),
          isTyping: false,
          currentStreamingMessageId: null
        }));
      }
    },

    // NEW: Set editing message
    setEditingMessage: (messageId) => {
      set({ editingMessageId: messageId });
    },

    // NEW: Delete message (and subsequent messages)
    deleteMessage: (messageId) => {
      const { messages } = get();
      const messageIndex = messages.findIndex(m => m.id === messageId);

      if (messageIndex === -1) return;

      // Remove the message and all messages after it
      set((state) => ({
        messages: state.messages.slice(0, messageIndex)
      }));
    }
  }))
);
