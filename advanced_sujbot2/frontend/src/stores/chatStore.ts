import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { ChatMessage, PipelineStatus, Source } from '@/types';
import { chatWebSocket } from '@/services/websocket';
import { generateId } from '@/utils/file';
import { useDocumentStore } from './documentStore';
import * as conversationApi from '@/services/conversationApi';

interface ChatStore {
  // Current conversation
  currentConversationId: string | null;
  messages: ChatMessage[];
  isTyping: boolean;
  currentStreamingMessageId: string | null;
  editingMessageId: string | null;

  // Conversation list (cached)
  conversations: conversationApi.ConversationListItem[];
  conversationsLoaded: boolean;

  // Actions - Message Management
  sendMessage: (text: string, documentIds?: string[]) => Promise<void>;
  addMessage: (message: ChatMessage) => void;
  updateStreamingMessage: (chunk: string) => void;
  updatePipelineStatus: (status: PipelineStatus) => void;
  attachSources: (sources: Source[]) => void;
  finishStreaming: () => void;
  clearChat: () => void;
  editMessage: (messageId: string, newText: string) => Promise<void>;
  regenerateResponse: (messageId: string) => Promise<void>;
  setEditingMessage: (messageId: string | null) => void;
  deleteMessage: (messageId: string) => void;

  // Actions - Conversation Management
  createNewConversation: () => Promise<void>;
  loadConversation: (conversationId: string) => Promise<void>;
  saveCurrentConversation: () => Promise<void>;
  listConversations: () => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<void>;
  archiveConversation: (conversationId: string) => Promise<void>;
  renameConversation: (conversationId: string, newTitle: string) => Promise<void>;
}

export const useChatStore = create<ChatStore>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        currentConversationId: null,
        messages: [],
        isTyping: false,
        currentStreamingMessageId: null,
        editingMessageId: null,
        conversations: [],
        conversationsLoaded: false,

        // ================================================================
        // Message Management
        // ================================================================

        sendMessage: async (text, documentIds) => {
          const userMessage: ChatMessage = {
            id: generateId(),
            type: 'user',
            content: text,
            timestamp: new Date()
          };

          // Add user message to UI
          set((state) => ({
            messages: [...state.messages, userMessage]
          }));

          // Save to database if conversation exists
          const { currentConversationId, messages } = get();
          if (currentConversationId) {
            try {
              await conversationApi.addMessage(currentConversationId, {
                type: 'user',
                content: text,
                sequence: messages.length - 1
              });
            } catch (error) {
              console.error('Failed to save user message:', error);
            }
          }

          // Create bot message placeholder
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
            // Get documents to use
            const documentStore = useDocumentStore.getState();
            const docsToUse = documentIds || documentStore.selectedDocumentIds;

            const finalDocumentIds = docsToUse.length > 0
              ? docsToUse
              : [
                  ...documentStore.contracts.filter(d => d.status === 'indexed'),
                  ...documentStore.laws.filter(d => d.status === 'indexed')
                ].map(d => d.id);

            // Send via WebSocket
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

        updatePipelineStatus: (status) => {
          const { currentStreamingMessageId } = get();
          if (!currentStreamingMessageId) return;

          set((state) => ({
            messages: state.messages.map(m =>
              m.id === currentStreamingMessageId
                ? { ...m, pipelineStatus: status }
                : m
            )
          }));
        },

        attachSources: (sources) => {
          const { currentStreamingMessageId } = get();
          if (!currentStreamingMessageId) return;

          set((state) => ({
            messages: state.messages.map(m =>
              m.id === currentStreamingMessageId
                ? { ...m, sources: sources }
                : m
            )
          }));
        },

        finishStreaming: async () => {
          const { currentStreamingMessageId, currentConversationId, messages } = get();
          if (!currentStreamingMessageId) return;

          // Find the bot message
          const botMessage = messages.find(m => m.id === currentStreamingMessageId);

          set((state) => ({
            messages: state.messages.map(m =>
              m.id === currentStreamingMessageId
                ? { ...m, isStreaming: false, pipelineStatus: undefined }
                : m
            ),
            isTyping: false,
            currentStreamingMessageId: null
          }));

          // Save bot message to database
          if (currentConversationId && botMessage) {
            try {
              await conversationApi.addMessage(currentConversationId, {
                type: 'bot',
                content: botMessage.content,
                sequence: messages.length - 1
              });
            } catch (error) {
              console.error('Failed to save bot message:', error);
            }
          }
        },

        addMessage: (message) => {
          set((state) => ({
            messages: [...state.messages, message]
          }));
        },

        clearChat: async () => {
          // Save current conversation before clearing
          await get().saveCurrentConversation();

          // Clear UI
          set({
            messages: [],
            currentConversationId: null
          });
        },

        editMessage: async (messageId, newText) => {
          const { messages } = get();
          const messageIndex = messages.findIndex(m => m.id === messageId);

          if (messageIndex === -1) return;

          // Remove all messages after the edited message
          const messagesToKeep = messages.slice(0, messageIndex);
          set({ messages: messagesToKeep, editingMessageId: null });

          // Save conversation before resending
          await get().saveCurrentConversation();

          // Resend the edited message
          await get().sendMessage(newText);
        },

        regenerateResponse: async (botMessageId) => {
          const { messages } = get();
          const botMessageIndex = messages.findIndex(m => m.id === botMessageId);

          if (botMessageIndex === -1 || botMessageIndex === 0) return;

          // Find the user message
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

          // Create new bot message
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
            const documentStore = useDocumentStore.getState();
            const docsToUse = documentStore.selectedDocumentIds;

            const finalDocumentIds = docsToUse.length > 0
              ? docsToUse
              : [
                  ...documentStore.contracts.filter(d => d.status === 'indexed'),
                  ...documentStore.laws.filter(d => d.status === 'indexed')
                ].map(d => d.id);

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

        setEditingMessage: (messageId) => {
          set({ editingMessageId: messageId });
        },

        deleteMessage: (messageId) => {
          const { messages } = get();
          const messageIndex = messages.findIndex(m => m.id === messageId);

          if (messageIndex === -1) return;

          set((state) => ({
            messages: state.messages.slice(0, messageIndex)
          }));
        },

        // ================================================================
        // Conversation Management
        // ================================================================

        createNewConversation: async () => {
          try {
            // Save current conversation if exists
            await get().saveCurrentConversation();

            // Create new conversation in database
            const conversation = await conversationApi.createConversation({
              title: 'New Conversation',
              document_ids: []
            });

            // Clear current messages and set new conversation
            set({
              currentConversationId: conversation.id,
              messages: [],
              isTyping: false,
              currentStreamingMessageId: null
            });

            // Refresh conversation list
            await get().listConversations();
          } catch (error) {
            console.error('Failed to create new conversation:', error);
          }
        },

        loadConversation: async (conversationId) => {
          try {
            // Save current conversation before switching
            await get().saveCurrentConversation();

            // Load conversation from database
            const conversation = await conversationApi.getConversation(conversationId);

            // Convert API messages to ChatMessage format
            const messages: ChatMessage[] = conversation.messages.map(msg => ({
              id: msg.id,
              type: msg.type,
              content: msg.content,
              timestamp: new Date(msg.created_at),
              isStreaming: false,
              // TODO: Add sources if available
            }));

            set({
              currentConversationId: conversation.id,
              messages,
              isTyping: false,
              currentStreamingMessageId: null
            });
          } catch (error) {
            console.error('Failed to load conversation:', error);
          }
        },

        saveCurrentConversation: async () => {
          const { currentConversationId, messages } = get();

          if (!currentConversationId || messages.length === 0) {
            return;
          }

          try {
            // Get existing conversation to check current message count
            const existingConv = await conversationApi.getConversation(currentConversationId);
            const existingMessageCount = existingConv.messages.length;

            // Only save new messages (not already in database)
            const newMessages = messages.slice(existingMessageCount);

            if (newMessages.length > 0) {
              // Save new messages
              const messagesToSave = newMessages.map((msg, idx) => ({
                type: msg.type,
                content: msg.content,
                sequence: existingMessageCount + idx,
                metadata: {}
              }));

              await conversationApi.addMessagesBatch(currentConversationId, messagesToSave);
            }

            // Refresh conversation list to update timestamps
            await get().listConversations();
          } catch (error) {
            console.error('Failed to save conversation:', error);
          }
        },

        listConversations: async () => {
          try {
            const response = await conversationApi.listConversations({
              limit: 50,
              offset: 0,
              include_archived: false
            });

            set({
              conversations: response.conversations,
              conversationsLoaded: true
            });
          } catch (error) {
            console.error('Failed to list conversations:', error);
          }
        },

        deleteConversation: async (conversationId) => {
          try {
            await conversationApi.deleteConversation(conversationId);

            // If deleting current conversation, clear it
            if (get().currentConversationId === conversationId) {
              set({
                currentConversationId: null,
                messages: []
              });
            }

            // Refresh conversation list
            await get().listConversations();
          } catch (error) {
            console.error('Failed to delete conversation:', error);
          }
        },

        archiveConversation: async (conversationId) => {
          try {
            await conversationApi.updateConversation(conversationId, {
              is_archived: true
            });

            // If archiving current conversation, clear it
            if (get().currentConversationId === conversationId) {
              set({
                currentConversationId: null,
                messages: []
              });
            }

            // Refresh conversation list
            await get().listConversations();
          } catch (error) {
            console.error('Failed to archive conversation:', error);
          }
        },

        renameConversation: async (conversationId, newTitle) => {
          try {
            await conversationApi.updateConversation(conversationId, {
              title: newTitle
            });

            // Refresh conversation list
            await get().listConversations();
          } catch (error) {
            console.error('Failed to rename conversation:', error);
          }
        }
      }),
      {
        name: 'chat-storage',
        partialize: (state) => ({
          currentConversationId: state.currentConversationId,
          // Don't persist messages - they're loaded from database
          // Don't persist conversations list - it's refreshed on load
        })
      }
    )
  )
);
