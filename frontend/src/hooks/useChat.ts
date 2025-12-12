/**
 * useChat hook - Manages chat state and SSE streaming
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { apiService } from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import type { Message, Conversation, ToolCall, ClarificationData } from '../types';

// UUID validation regex for conversation IDs
const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * Validate and extract conversation ID from URL parameter
 */
function getValidConversationIdFromUrl(): string | null {
  if (typeof window === 'undefined') return null;
  const params = new URLSearchParams(window.location.search);
  const urlConversationId = params.get('c');
  // Only accept valid UUID format to prevent invalid state
  if (urlConversationId && UUID_REGEX.test(urlConversationId)) {
    return urlConversationId;
  }
  return null;
}

// Spending limit error data from 402 response
export interface SpendingLimitError {
  message_cs: string;
  message_en: string;
  total_spent_czk: number;
  spending_limit_czk: number;
}

export function useChat() {
  const { isAuthenticated } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  // Current conversation ID - initialized from URL query param for refresh persistence
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(
    getValidConversationIdFromUrl
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const [clarificationData, setClarificationData] = useState<ClarificationData | null>(null);
  const [awaitingClarification, setAwaitingClarification] = useState(false);
  // Spending tracking
  const [spendingLimitError, setSpendingLimitError] = useState<SpendingLimitError | null>(null);
  const [spendingRefreshTrigger, setSpendingRefreshTrigger] = useState(0);

  // Sync URL with current conversation ID (for refresh persistence and shareable links)
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const url = new URL(window.location.href);
    if (currentConversationId) {
      url.searchParams.set('c', currentConversationId);
    } else {
      url.searchParams.delete('c');
    }

    // Update URL without page reload (replaceState to avoid history spam)
    window.history.replaceState({}, '', url.toString());
  }, [currentConversationId]);

  // Handle browser back/forward navigation
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handlePopState = () => {
      // Use same validation as initial load
      setCurrentConversationId(getValidConversationIdFromUrl());
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Load conversations from server when user is authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      // Clear conversations on logout
      setConversations([]);
      setCurrentConversationId(null);
      return;
    }

    const loadConversations = async () => {
      try {
        const serverConversations = await apiService.getConversations();
        setConversations(serverConversations);
      } catch (error) {
        console.error('Failed to load conversations from server:', error);
        // Don't block UI - user can still create new conversations
      }
    };

    loadConversations();
  }, [isAuthenticated]); // Re-run when auth state changes (login/logout)

  // Validate that current conversation still exists after loading conversations
  useEffect(() => {
    if (currentConversationId && conversations.length > 0) {
      const conversationExists = conversations.some(c => c.id === currentConversationId);
      if (!conversationExists) {
        // Current conversation was deleted, clear it
        console.log('Current conversation no longer exists, clearing currentConversationId');
        setCurrentConversationId(null);
      }
    }
  }, [conversations, currentConversationId]);

  // Refs for managing streaming state
  const currentMessageRef = useRef<Message | null>(null);
  const currentToolCallsRef = useRef<Map<string, ToolCall>>(new Map());

  // AbortController for cancelling streaming on page refresh/navigation
  // This ensures backend doesn't continue processing when user leaves the page
  const abortControllerRef = useRef<AbortController | null>(null);

  // Warn user and cancel streaming on page unload (refresh, close, navigation)
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      // Only show warning if actively streaming
      if (abortControllerRef.current) {
        // Standard way to show browser's "Leave page?" dialog
        e.preventDefault();
        // For older browsers (Chrome < 119)
        e.returnValue = '';

        console.log('🔄 useChat: Aborting stream due to page unload');
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      // Also cleanup on hook unmount
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, []);

  // Ref pattern to avoid stale closures in async callbacks
  //
  // Problem: In regenerateMessage and editMessage, we read conversation state from an async callback
  // that may execute after state updates. Using `conversations` directly would capture
  // the value from when the callback was created (stale closure).
  //
  // Solution: Keep a ref that always points to latest conversations array. The ref
  // object never changes identity, so it's safe to read in async contexts.
  //
  // When to use: Async callbacks that need latest state (regenerate, edit)
  // When NOT to use: Synchronous operations - use functional setState instead
  const conversationsRef = useRef<Conversation[]>(conversations);

  // Keep ref synchronized with state
  useEffect(() => {
    conversationsRef.current = conversations;
  }, [conversations]);

  /**
   * Clean invalid/incomplete messages from conversation
   * Prevents corrupted data in database
   */
  const cleanMessages = useCallback((messages: Message[]): Message[] => {
    return messages.filter(msg =>
      msg &&
      msg.role &&
      msg.id &&
      msg.timestamp &&
      msg.content !== undefined
    );
  }, []);

  /**
   * Get current conversation
   */
  const currentConversation = conversations.find((c) => c.id === currentConversationId);

  /**
   * Create a new conversation
   * Uses pushState for browser history (back/forward navigation)
   */
  const createConversation = useCallback(async () => {
    try {
      // Create conversation on server
      const newConversation = await apiService.createConversation('New Conversation');

      // Update local state
      setConversations((prev) => [...prev, newConversation]);
      setCurrentConversationId(newConversation.id);

      // Add to browser history for back/forward navigation
      const url = new URL(window.location.href);
      url.searchParams.set('c', newConversation.id);
      window.history.pushState({}, '', url.toString());

      return newConversation;
    } catch (error) {
      console.error('Failed to create conversation:', error);
      throw error;
    }
  }, []);

  /**
   * Select a conversation
   * Uses pushState for browser history (back/forward navigation)
   */
  const selectConversation = useCallback((id: string) => {
    // Only update if different conversation
    if (id === currentConversationId) return;

    // Update state
    setCurrentConversationId(id);

    // Add to browser history for back/forward navigation
    const url = new URL(window.location.href);
    url.searchParams.set('c', id);
    window.history.pushState({}, '', url.toString());
  }, [currentConversationId]);

  /**
   * Load messages when conversation is selected
   * This ensures messages are fetched from database after page refresh
   */
  useEffect(() => {
    if (!currentConversationId) return;

    // Wait for conversations to load before fetching messages
    // This prevents unnecessary API calls for invalid/deleted conversation IDs
    if (conversations.length === 0 && isAuthenticated) return;

    // Verify conversation exists before loading messages
    const conversationExists = conversations.some(c => c.id === currentConversationId);
    if (!conversationExists && conversations.length > 0) return;

    const loadMessages = async () => {
      try {
        const messages = await apiService.getConversationHistory(currentConversationId);

        setConversations((prev) =>
          prev.map((conv) =>
            conv.id === currentConversationId
              ? { ...conv, messages }
              : conv
          )
        );
      } catch (error) {
        console.error('Failed to load conversation messages:', error);
      }
    };

    loadMessages();
  }, [currentConversationId, conversations.length, isAuthenticated]);

  /**
   * Delete a conversation
   */
  const deleteConversation = useCallback(async (id: string) => {
    try {
      // Delete from server
      await apiService.deleteConversation(id);

      // Update local state
      setConversations((prev) => prev.filter((c) => c.id !== id));

      if (currentConversationId === id) {
        setCurrentConversationId(null);
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
      // Still remove from UI even if server delete fails (eventual consistency)
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (currentConversationId === id) {
        setCurrentConversationId(null);
      }
    }
  }, [currentConversationId]);

  /**
   * Send a message and stream the response
   * @param content - The message content to send
   * @param addUserMessage - Whether to add a new user message (false for regenerate/edit)
   * @param selectedContext - Optional selected text from PDF for additional context
   */
  const sendMessage = useCallback(
    async (content: string, addUserMessage: boolean = true, selectedContext?: {
      text: string;
      documentId: string;
      documentName: string;
      pageStart: number;
      pageEnd: number;
    } | null) => {
      if (isStreaming || !content.trim()) {
        return;
      }

      // Get current conversation using currentConversationId
      let conversation: Conversation;

      if (currentConversationId) {
        // Find conversation from REF (not state) to ensure we have latest version
        // This is critical for edit/regenerate where state might have been updated
        // in the same tick (truncation) but not yet committed to the 'conversations' variable
        const found = conversationsRef.current.find((c) => c.id === currentConversationId);
        if (found) {
          conversation = found;
        } else {
          conversation = await createConversation();
        }
      } else {
        conversation = await createConversation();
      }

      let updatedConversation: Conversation;

      if (addUserMessage) {
        // Create and add user message
        const userMessage: Message = {
          id: `msg_${Date.now()}_user`,
          role: 'user',
          content: content.trim(),
          timestamp: new Date().toISOString(),
        };

        // Add user message to conversation
        updatedConversation = {
          ...conversation,
          messages: [...conversation.messages, userMessage],
          updatedAt: new Date().toISOString(),
          title: conversation.title,  // Let backend generate LLM title
        };
      } else {
        // Regenerate/edit mode - use conversation as-is
        updatedConversation = {
          ...conversation,
          updatedAt: new Date().toISOString(),
        };
      }

      setConversations((prev) =>
        prev.map((c) => (c.id === updatedConversation.id ? updatedConversation : c))
      );

      // Initialize assistant message with agent progress
      currentMessageRef.current = {
        id: `msg_${Date.now()}_assistant`,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        toolCalls: [],
        agentProgress: {
          currentAgent: 'orchestrator', // Start with orchestrator immediately
          currentMessage: 'Initializing...',
          completedAgents: [],
          activeTools: [],
          isStreaming: true // Keep indicator visible until done event
        }
      };
      currentToolCallsRef.current = new Map();

      // IMMEDIATE UPDATE: Add placeholder message to state so UI shows "Thinking..." instantly
      // This prevents the "empty gap" before first backend event arrives
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== updatedConversation.id) return c;
          return {
            ...c,
            messages: [...c.messages, currentMessageRef.current!]
          };
        })
      );

      setIsStreaming(true);

      // Create new AbortController for this stream
      // Abort any previous stream first (shouldn't happen but defensive)
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      try {
        // Prepare message history for context (last 10 pairs = 20 messages)
        // Exclude the current user message if we just added it to avoid duplication
        const existingMessages = addUserMessage
          ? updatedConversation.messages.slice(0, -1)  // Exclude just-added user message
          : updatedConversation.messages;
        const historyLimit = 20; // Last 10 user+assistant pairs
        const messageHistory = existingMessages
          .slice(-historyLimit)
          .filter(msg => msg.role === 'user' || msg.role === 'assistant')
          .map(msg => ({
            role: msg.role as 'user' | 'assistant',
            content: msg.content
          }));

        // Format selected context for API (convert camelCase to snake_case)
        const apiSelectedContext = selectedContext ? {
          text: selectedContext.text,
          document_id: selectedContext.documentId,
          document_name: selectedContext.documentName,
          page_start: selectedContext.pageStart,
          page_end: selectedContext.pageEnd,
        } : null;

        // Stream response from backend with abort signal for page refresh handling
        for await (const event of apiService.streamChat(
          content,
          conversation.id,
          !addUserMessage,  // Skip saving user message when regenerating/editing (already exists)
          messageHistory,   // Pass conversation history for context
          abortControllerRef.current.signal,  // Allow cancellation on page refresh
          apiSelectedContext  // Pass selected text from PDF for additional context
        )) {
          // Handle tool health check (first event)
          if (event.event === 'tool_health') {
            // Log tool health status (visible in browser console)
            if (!event.data.healthy) {
              console.warn('Tool health warning:', event.data.summary);
            } else {
              console.info('Tool health:', event.data.summary);
            }
            // Store tool health in message metadata for debugging
            if (currentMessageRef.current) {
              currentMessageRef.current.toolHealth = {
                healthy: event.data.healthy,
                summary: event.data.summary,
                unavailableTools: event.data.unavailable_tools,
                degradedTools: event.data.degraded_tools,
              };
            }
          }
          // Handle agent progress events
          else if (event.event === 'agent_start') {
            if (currentMessageRef.current && currentMessageRef.current.agentProgress) {
              // Mark previous agent as completed
              if (currentMessageRef.current.agentProgress.currentAgent) {
                currentMessageRef.current.agentProgress.completedAgents.push(
                  currentMessageRef.current.agentProgress.currentAgent
                );
              }

              // Set new current agent and clear activeTools
              currentMessageRef.current.agentProgress.currentAgent = event.data.agent;
              currentMessageRef.current.agentProgress.currentMessage = event.data.message;
              currentMessageRef.current.agentProgress.activeTools = [];

              // Update UI - IMPORTANT: Deep copy agentProgress to trigger React re-render
              // Capture snapshot of current message to avoid race conditions where ref becomes null
              const currentMessageSnapshot = { ...currentMessageRef.current };
              const currentAgentProgressSnapshot = currentMessageRef.current.agentProgress
                ? {
                  ...currentMessageRef.current.agentProgress,
                  activeTools: [...(currentMessageRef.current.agentProgress.activeTools || [])],
                  completedAgents: [...(currentMessageRef.current.agentProgress.completedAgents || [])]
                }
                : undefined;

              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  const lastMsg = messages[messages.length - 1];

                  // Deep copy message with agentProgress to ensure React detects changes
                  const updatedMessage = {
                    ...currentMessageSnapshot,
                    agentProgress: currentAgentProgressSnapshot
                  };

                  if (lastMsg?.role === 'assistant') {
                    messages[messages.length - 1] = updatedMessage;
                  } else {
                    messages.push(updatedMessage);
                  }

                  return { ...c, messages };
                })
              );
            }
          }
          else if (event.event === 'tool_call') {
            // Tool call event (running/completed/failed)
            if (currentMessageRef.current && currentMessageRef.current.agentProgress) {
              const { tool, status, timestamp } = event.data;

              if (status === 'running') {
                // Add new tool to activeTools
                currentMessageRef.current.agentProgress.activeTools.push({
                  tool,
                  status,
                  timestamp
                });
              } else if (status === 'completed' || status === 'failed') {
                // Update status of existing tool
                const toolIndex = currentMessageRef.current.agentProgress.activeTools.findIndex(
                  t => t.tool === tool && t.status === 'running'
                );
                if (toolIndex >= 0) {
                  currentMessageRef.current.agentProgress.activeTools[toolIndex].status = status;
                }
              }

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  const lastMsg = messages[messages.length - 1];

                  if (lastMsg?.role === 'assistant') {
                    messages[messages.length - 1] = { ...currentMessageRef.current! };
                  } else {
                    messages.push({ ...currentMessageRef.current! });
                  }

                  return { ...c, messages };
                })
              );
            }
          }
          else if (event.event === 'text_delta') {
            // Append text delta
            if (currentMessageRef.current) {
              // Transition to synthesizing phase when text starts arriving
              if (currentMessageRef.current.agentProgress && currentMessageRef.current.agentProgress.currentAgent && currentMessageRef.current.agentProgress.currentAgent !== 'synthesizing') {
                currentMessageRef.current.agentProgress.completedAgents.push(
                  currentMessageRef.current.agentProgress.currentAgent
                );
                currentMessageRef.current.agentProgress.currentAgent = 'synthesizing';
                currentMessageRef.current.agentProgress.currentMessage = null;
                currentMessageRef.current.agentProgress.activeTools = [];
              }

              currentMessageRef.current.content += event.data.content;

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  const lastMsg = messages[messages.length - 1];

                  if (lastMsg?.role === 'assistant') {
                    // Update existing assistant message
                    messages[messages.length - 1] = { ...currentMessageRef.current! };
                  } else {
                    // Add new assistant message
                    messages.push({ ...currentMessageRef.current! });
                  }

                  return { ...c, messages };
                })
              );
            }
          }
          else if (event.event === 'tool_result') {
            // Tool execution completed
            const callId = event.data.call_id;
            const existingToolCall = currentToolCallsRef.current.get(callId);

            if (existingToolCall) {
              existingToolCall.result = event.data.result;
              existingToolCall.executionTimeMs = event.data.execution_time_ms;
              existingToolCall.success = event.data.success;
              existingToolCall.status = event.data.success ? 'completed' : 'failed';

              if (currentMessageRef.current) {
                currentMessageRef.current.toolCalls = Array.from(
                  currentToolCallsRef.current.values()
                );
              }

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  messages[messages.length - 1] = { ...currentMessageRef.current! };

                  return { ...c, messages };
                })
              );
            } else {
              // Tool result without matching tool call (race condition or backend error)
              console.error('❌ Received tool_result for unknown call_id:', {
                callId,
                knownCallIds: Array.from(currentToolCallsRef.current.keys()),
                resultData: event.data
              });

              // Add warning message to chat
              if (currentMessageRef.current) {
                currentMessageRef.current.content +=
                  `\n\n[Warning: Received result for tool call ${callId} but call not found]`;
              }
            }
          } else if (event.event === 'tool_calls_summary') {
            // Tool calls summary from backend
            if (currentMessageRef.current && event.data.tool_calls) {
              // Convert backend tool calls to frontend format
              currentMessageRef.current.toolCalls = event.data.tool_calls.map((tc: any) => ({
                id: tc.id,
                name: tc.name,
                input: tc.input,
                result: tc.result,
                executionTimeMs: tc.executionTimeMs,
                success: tc.success,
                status: tc.success === false ? 'failed' as const : 'completed' as const,
                explicitParams: tc.explicitParams,
              }));

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  messages[messages.length - 1] = { ...currentMessageRef.current! };

                  return { ...c, messages };
                })
              );
            }
          } else if (event.event === 'cost_summary') {
            // Cost tracking summary with per-agent breakdown
            console.log('💰 FRONTEND: Cost summary received:', event.data);

            if (currentMessageRef.current) {
              // Validate and sanitize cost data
              const totalCost = typeof event.data.total_cost === 'number' ? event.data.total_cost : 0;
              const inputTokens = typeof event.data.total_input_tokens === 'number' ? event.data.total_input_tokens : 0;
              const outputTokens = typeof event.data.total_output_tokens === 'number' ? event.data.total_output_tokens : 0;
              const agentBreakdown = Array.isArray(event.data.agent_breakdown) ? event.data.agent_breakdown : [];

              // Log validation warnings
              if (totalCost < 0 || isNaN(totalCost)) {
                console.error('❌ Invalid cost value received:', event.data.total_cost);
              }
              if (inputTokens < 0 || outputTokens < 0) {
                console.error('❌ Invalid token counts:', { inputTokens, outputTokens });
              }

              currentMessageRef.current.cost = {
                totalCost: Math.max(0, totalCost),
                inputTokens: Math.max(0, inputTokens),
                outputTokens: Math.max(0, outputTokens),
                cachedTokens: event.data.cache_stats?.cache_read_tokens || 0,
                agentBreakdown,
                cacheStats: event.data.cache_stats || { cache_read_tokens: 0, cache_creation_tokens: 0 },
              };

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  messages[messages.length - 1] = { ...currentMessageRef.current! };

                  return { ...c, messages };
                })
              );
            }
          } else if (event.event === 'clarification_needed') {
            // HITL: Agent needs clarification
            console.log('🤔 FRONTEND: Clarification needed:', event.data);

            setClarificationData(event.data);
            setAwaitingClarification(true);

            // Don't emit "done" - workflow is paused
            return;
          } else if (event.event === 'title_update') {
            // Title was generated by LLM for new conversation
            const newTitle = event.data?.title;
            if (newTitle) {
              setConversations((prev) =>
                prev.map((c) =>
                  c.id === updatedConversation.id ? { ...c, title: newTitle } : c
                )
              );
            }
          } else if (event.event === 'done') {
            // Stream completed - mark streaming as finished
            if (currentMessageRef.current?.agentProgress) {
              currentMessageRef.current.agentProgress.isStreaming = false;
              currentMessageRef.current.agentProgress.currentAgent = null;
            }
            break;
          } else if (event.event === 'error') {
            // Error occurred
            console.error('Stream error:', event.data);

            // Handle spending limit exceeded error specially
            if (event.data.type === 'SpendingLimitExceeded') {
              setSpendingLimitError({
                message_cs: event.data.message_cs || 'Byl dosažen limit výdajů.',
                message_en: event.data.message_en || 'Spending limit reached.',
                total_spent_czk: event.data.total_spent_czk || 0,
                spending_limit_czk: event.data.spending_limit_czk || 0,
              });
              // Don't add error to message - show modal instead
              break;
            }

            if (currentMessageRef.current) {
              currentMessageRef.current.content += `\n\n[Error: ${event.data.error}]`;
            }
          }
        }

        // Ensure final message is in state before saving
        // Capture the ref value before async operations
        const finalMessage = currentMessageRef.current;
        if (finalMessage) {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== updatedConversation.id) return c;

              const messages = [...c.messages];
              const lastMsg = messages[messages.length - 1];

              // Update or add final assistant message
              if (lastMsg?.role === 'assistant') {
                messages[messages.length - 1] = { ...finalMessage };
              } else {
                messages.push({ ...finalMessage });
              }

              // Clean messages - remove any incomplete/invalid messages
              const cleanedMessages = messages.filter(msg =>
                msg &&
                msg.role &&
                msg.id &&
                msg.timestamp &&
                msg.content !== undefined
              );

              // Sync messageCount with actual messages length
              const finalConv = {
                ...c,
                messages: cleanedMessages,
                messageCount: cleanedMessages.length,
                updatedAt: new Date().toISOString()
              };

              return finalConv;
            })
          );
        }
      } catch (error) {
        // Check if this was an intentional abort (page refresh)
        if (error instanceof Error && error.name === 'AbortError') {
          console.log('🔄 useChat: Stream aborted (page refresh or navigation)');
          // Don't show error to user - this is expected behavior
        } else {
          console.error('❌ Error during streaming:', error);
          console.error('Error stack:', (error as Error).stack);

          // Add error message
          if (currentMessageRef.current) {
            currentMessageRef.current.content += `\n\n[Error: ${(error as Error).message}]`;
          }
        }
      } finally {
        setIsStreaming(false);
        currentMessageRef.current = null;
        currentToolCallsRef.current = new Map();
        abortControllerRef.current = null;  // Cleanup abort controller
        // Invalidate spending cache and trigger refresh after each message
        apiService.invalidateSpendingCache();
        setSpendingRefreshTrigger((prev) => prev + 1);
      }
    },
    [isStreaming, createConversation, currentConversationId, conversations]
  );

  /**
   * Edit a user message and resend
   */
  const editMessage = useCallback(
    async (messageId: string, newContent: string) => {
      if (isStreaming || !newContent.trim()) return;

      // Use functional update to get current conversation state
      let shouldSend = false;
      let messageIndex = -1;
      let conversationId = '';

      setConversations((prev) => {
        // Find the current conversation
        const currentConv = prev.find((c) => c.id === currentConversationId);
        if (!currentConv) return prev;

        // Find the message index
        messageIndex = currentConv.messages.findIndex((m) => m.id === messageId);
        if (messageIndex === -1 || currentConv.messages[messageIndex].role !== 'user') {
          return prev;
        }

        conversationId = currentConv.id;
        shouldSend = true;

        // Remove all messages after this one (including assistant response)
        const updatedMessages = currentConv.messages.slice(0, messageIndex);

        // Clean messages before saving
        const cleanedMessages = cleanMessages(updatedMessages);

        // Update conversation with truncated messages
        const updatedConversation: Conversation = {
          ...currentConv,
          messages: cleanedMessages,
          updatedAt: new Date().toISOString(),
        };

        // Return updated conversations array
        return prev.map((c) => (c.id === currentConversationId ? updatedConversation : c));
      });

      // If validation passed, delete from backend and send the edited message
      if (shouldSend) {
        // Delete from backend in background (don't await - fire and forget for immediate UI response)
        apiService.truncateMessagesAfter(conversationId, messageIndex).catch((error) => {
          console.error('Failed to truncate messages in database:', error);
          // Non-critical - frontend state is already updated
        });

        // Race condition fix: Wait for React to commit state update before sending
        // Without this, sendMessage may read stale conversation state (pre-update)
        // causing duplicate messages or incorrect context.
        //
        // Why 10ms: Minimum delay that ensures React has flushed state to DOM in all browsers
        // (0ms would queue immediately, potentially before state commit in concurrent mode)
        //
        // Alternative considered: useEffect with dependency, but creates unnecessary rerender
        await new Promise(resolve => setTimeout(resolve, 10));
        await sendMessage(newContent);
      }
    },
    [isStreaming, sendMessage, currentConversationId]
  );

  /**
   * Regenerate the last assistant response
   */
  const regenerateMessage = useCallback(
    async (messageId: string) => {
      if (isStreaming) {
        return;
      }

      // SYNCHRONOUSLY read current conversation from ref (always fresh)
      const currentConv = conversationsRef.current.find((c) => c.id === currentConversationId);

      if (!currentConv) {
        return;
      }

      // Find the message index
      const messageIndex = currentConv.messages.findIndex((m) => m.id === messageId);

      if (messageIndex === -1) {
        return;
      }

      const message = currentConv.messages[messageIndex];

      if (message.role !== 'assistant') {
        return;
      }

      // Find the user message before this assistant message
      const userMessageIndex = messageIndex - 1;

      if (userMessageIndex < 0) {
        return;
      }

      const userMessage = currentConv.messages[userMessageIndex];

      if (userMessage.role !== 'user') {
        return;
      }

      const userMessageContent = userMessage.content;
      const conversationId = currentConv.id;

      // Remove all messages after (and including) the assistant message we want to regenerate
      const updatedMessages = currentConv.messages.slice(0, messageIndex);

      // Clean messages before saving
      const cleanedMessages = cleanMessages(updatedMessages);

      // Update conversation
      const updatedConversation: Conversation = {
        ...currentConv,
        messages: cleanedMessages,
        updatedAt: new Date().toISOString(),
      };

      // IMMEDIATELY update state to remove message from UI
      setConversations((prev) =>
        prev.map((c) => (c.id === conversationId ? updatedConversation : c))
      );

      // Delete from backend in background (don't await - fire and forget for immediate UI response)
      apiService.truncateMessagesAfter(conversationId, messageIndex).catch((error) => {
        console.error('Failed to truncate messages in database:', error);
        // Non-critical - frontend state is already updated
      });

      // Race condition fix: Wait for React to commit state update before sending
      // Without this, sendMessage may read stale conversation state (pre-truncation)
      // causing duplicate messages or incorrect context.
      //
      // Why 10ms: Minimum delay that ensures React has flushed state to DOM in all browsers
      // (0ms would queue immediately, potentially before state commit in concurrent mode)
      //
      // Alternative considered: useEffect with dependency, but creates unnecessary rerender
      await new Promise(resolve => setTimeout(resolve, 10));
      // Pass false to prevent adding a new user message (we're regenerating from existing)
      await sendMessage(userMessageContent, false);
    },
    [isStreaming, sendMessage, currentConversationId, cleanMessages]
  );

  /**
   * Submit clarification response and resume workflow
   */
  const submitClarification = useCallback(
    async (response: string) => {
      if (!clarificationData || !response.trim()) {
        return;
      }

      const { thread_id } = clarificationData;

      // Reset clarification state
      setClarificationData(null);
      setAwaitingClarification(false);

      // Get current conversation
      const conversation = conversations.find((c) => c.id === currentConversationId);
      if (!conversation) {
        console.error('No current conversation found');
        return;
      }

      // Initialize assistant message for resumed workflow
      currentMessageRef.current = {
        id: `msg_${Date.now()}_assistant`,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        toolCalls: [],
      };
      currentToolCallsRef.current = new Map();

      setIsStreaming(true);

      try {
        // Stream clarification response from backend
        for await (const event of apiService.streamClarification(thread_id, response)) {
          console.log('📨 FRONTEND: Received clarification event:', event.event);

          if (event.event === 'text_delta') {
            // Append text delta
            if (currentMessageRef.current) {
              currentMessageRef.current.content += event.data.content;

              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== conversation.id) return c;

                  const messages = [...c.messages];
                  const lastMsg = messages[messages.length - 1];

                  if (lastMsg?.role === 'assistant') {
                    messages[messages.length - 1] = { ...currentMessageRef.current! };
                  } else {
                    messages.push({ ...currentMessageRef.current! });
                  }

                  return { ...c, messages };
                })
              );
            }
          } else if (event.event === 'cost_summary') {
            // Cost tracking update
            if (currentMessageRef.current) {
              currentMessageRef.current.cost = {
                totalCost: event.data.total_cost,
                inputTokens: event.data.input_tokens,
                outputTokens: event.data.output_tokens,
                cachedTokens: event.data.cached_tokens,
                agentBreakdown: event.data.agent_breakdown,
              };

              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== conversation.id) return c;

                  const messages = [...c.messages];
                  messages[messages.length - 1] = { ...currentMessageRef.current! };

                  return { ...c, messages };
                })
              );
            }
          } else if (event.event === 'done') {
            // Stream completed - mark streaming as finished
            if (currentMessageRef.current?.agentProgress) {
              currentMessageRef.current.agentProgress.isStreaming = false;
              currentMessageRef.current.agentProgress.currentAgent = null;
            }
            break;
          } else if (event.event === 'error') {
            // Error occurred
            console.error('Clarification stream error:', event.data);

            if (currentMessageRef.current) {
              currentMessageRef.current.content += `\n\n[Error: ${event.data.error}]`;
            }
          }
        }

        // Save final message
        const finalMessage = currentMessageRef.current;
        if (finalMessage) {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== conversation.id) return c;

              const messages = [...c.messages];
              const lastMsg = messages[messages.length - 1];

              if (lastMsg?.role === 'assistant') {
                messages[messages.length - 1] = { ...finalMessage };
              } else {
                messages.push({ ...finalMessage });
              }

              const cleanedMessages = cleanMessages(messages);
              // Sync messageCount with actual messages length
              const finalConv = {
                ...c,
                messages: cleanedMessages,
                messageCount: cleanedMessages.length,
                updatedAt: new Date().toISOString()
              };

              return finalConv;
            })
          );
        }
      } catch (error) {
        console.error('❌ Error during clarification streaming:', error);

        if (currentMessageRef.current) {
          currentMessageRef.current.content += `\n\n[Error: ${(error as Error).message}]`;
        }
      } finally {
        setIsStreaming(false);
        currentMessageRef.current = null;
        currentToolCallsRef.current = new Map();
      }
    },
    [clarificationData, conversations, currentConversationId, cleanMessages]
  );

  /**
   * Cancel clarification and continue with original query
   */
  const cancelClarification = useCallback(() => {
    console.log('🚫 FRONTEND: Clarification cancelled - continuing with original query');
    setClarificationData(null);
    setAwaitingClarification(false);
    setIsStreaming(false);

    // Optionally: Could show a message to the user that clarification was skipped
  }, []);

  /**
   * Rename a conversation
   * @param id - Conversation ID
   * @param newTitle - New title for the conversation
   */
  const renameConversation = useCallback(
    async (id: string, newTitle: string) => {
      if (!newTitle.trim()) {
        console.warn('Cannot rename conversation to empty title');
        return;
      }

      try {
        // Update on server
        await apiService.updateConversationTitle(id, newTitle.trim());

        // Update local state
        setConversations((prev) =>
          prev.map((c) =>
            c.id === id ? { ...c, title: newTitle.trim(), updatedAt: new Date().toISOString() } : c
          )
        );
      } catch (error) {
        console.error('Failed to rename conversation:', error);
        // Don't throw - UI should handle gracefully
      }
    },
    []
  );

  /**
   * Delete a message from the current conversation
   * @param messageId - ID of the message to delete
   */
  const deleteMessage = useCallback(
    async (messageId: string) => {
      if (!currentConversationId) {
        console.warn('No conversation selected');
        return;
      }

      // Try backend notification (non-blocking)
      try {
        await apiService.deleteMessage(currentConversationId, messageId);
      } catch (error) {
        console.error('Backend delete notification failed:', error);
      }

      // Always update UI (frontend-centric approach)
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== currentConversationId) return c;

          const updatedMessages = c.messages.filter((m) => m.id !== messageId);
          const updatedConv = {
            ...c,
            messages: updatedMessages,
            updatedAt: new Date().toISOString(),
          };

          return updatedConv;
        })
      );

      console.log(`✓ Message ${messageId} deleted locally`);
    },
    [currentConversationId]
  );

  /**
   * Cancel ongoing streaming
   * Aborts the current stream and cleans up state
   */
  const cancelStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      console.log('🛑 useChat: User cancelled streaming');
      abortControllerRef.current.abort();
      abortControllerRef.current = null;

      // Clean up streaming state
      setIsStreaming(false);
      currentMessageRef.current = null;
      currentToolCallsRef.current = new Map();
    }
  }, []);

  /**
   * Clear spending limit error (dismiss modal)
   */
  const clearSpendingLimitError = useCallback(() => {
    setSpendingLimitError(null);
  }, []);

  return {
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
    deleteMessage,
    submitClarification,
    cancelClarification,
    cancelStreaming,
    clearSpendingLimitError,
  };
}
