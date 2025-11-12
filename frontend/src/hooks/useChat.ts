/**
 * useChat hook - Manages chat state and SSE streaming
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { apiService } from '../services/api';
import { storageService } from '../lib/storage';
import type { Message, Conversation, ToolCall, ClarificationData } from '../types';

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>(() =>
    storageService.getConversations()
  );
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(() =>
    storageService.getCurrentConversationId()
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>(() =>
    // Load from localStorage, or use Gemini 2.5 Flash Lite as default
    storageService.getSelectedModel() || 'gemini-2.5-flash-latest-exp-1206'
  );
  const [clarificationData, setClarificationData] = useState<ClarificationData | null>(null);
  const [awaitingClarification, setAwaitingClarification] = useState(false);

  // Refs for managing streaming state
  const currentMessageRef = useRef<Message | null>(null);
  const currentToolCallsRef = useRef<Map<string, ToolCall>>(new Map());

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

  // Initialize and verify model
  useEffect(() => {
    const savedModel = storageService.getSelectedModel();

    // If no saved model, save the default
    if (!savedModel) {
      storageService.setSelectedModel('gemini-2.5-flash-latest-exp-1206');
    }

    // Verify model is available on backend
    apiService.getModels().then((data) => {
      const currentModel = savedModel || 'gemini-2.5-flash-latest-exp-1206';

      // If current model is not available, fallback to backend default
      if (!data.models.some((m: any) => m.id === currentModel)) {
        const defaultModel = data.defaultModel || 'gemini-2.5-flash-latest-exp-1206';
        setSelectedModel(defaultModel);
        storageService.setSelectedModel(defaultModel);
      }
    }).catch(console.error);
  }, []);

  /**
   * Clean invalid/incomplete messages from conversation
   * Prevents corrupted data in localStorage
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
   */
  const createConversation = useCallback(() => {
    const newConversation: Conversation = {
      id: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title: 'New Conversation',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    setConversations((prev) => [...prev, newConversation]);
    setCurrentConversationId(newConversation.id);
    storageService.saveConversation(newConversation);
    storageService.setCurrentConversationId(newConversation.id);

    return newConversation;
  }, []);

  /**
   * Select a conversation
   */
  const selectConversation = useCallback((id: string) => {
    setCurrentConversationId(id);
    storageService.setCurrentConversationId(id);
  }, []);

  /**
   * Delete a conversation
   */
  const deleteConversation = useCallback((id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    storageService.deleteConversation(id);

    if (currentConversationId === id) {
      setCurrentConversationId(null);
    }
  }, [currentConversationId]);

  /**
   * Send a message and stream the response
   * @param content - The message content to send
   * @param addUserMessage - Whether to add a new user message (false for regenerate/edit)
   */
  const sendMessage = useCallback(
    async (content: string, addUserMessage: boolean = true) => {
      if (isStreaming || !content.trim()) {
        return;
      }

      // Get current conversation using currentConversationId
      let conversation: Conversation;

      if (currentConversationId) {
        // Find conversation from state using functional read
        const found = conversations.find((c) => c.id === currentConversationId);
        if (found) {
          conversation = found;
        } else {
          conversation = createConversation();
        }
      } else {
        conversation = createConversation();
      }

      let updatedConversation: Conversation;

      if (addUserMessage) {
        // Create and add user message
        const userMessage: Message = {
          id: `msg_${Date.now()}_user`,
          role: 'user',
          content: content.trim(),
          timestamp: new Date(),
        };

        // Add user message to conversation
        updatedConversation = {
          ...conversation,
          messages: [...conversation.messages, userMessage],
          updatedAt: new Date(),
          title: conversation.messages.length === 0 ? content.slice(0, 50) : conversation.title,
        };
      } else {
        // Regenerate/edit mode - use conversation as-is
        updatedConversation = {
          ...conversation,
          updatedAt: new Date(),
        };
      }

      setConversations((prev) =>
        prev.map((c) => (c.id === updatedConversation.id ? updatedConversation : c))
      );
      storageService.saveConversation(updatedConversation);

      // Initialize assistant message with agent progress
      currentMessageRef.current = {
        id: `msg_${Date.now()}_assistant`,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        toolCalls: [],
        agentProgress: {
          currentAgent: null,
          currentMessage: null,
          completedAgents: [],
          activeTools: []
        }
      };
      currentToolCallsRef.current = new Map();

      setIsStreaming(true);

      try {
        // Stream response from backend
        for await (const event of apiService.streamChat(
          content,
          conversation.id,
          selectedModel
        )) {
          console.log('📨 FRONTEND: Received event:', event.event);

          // Handle agent progress events
          if (event.event === 'agent_start') {
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
              // Mark all agents as completed when text starts arriving
              if (currentMessageRef.current.agentProgress && currentMessageRef.current.agentProgress.currentAgent) {
                currentMessageRef.current.agentProgress.completedAgents.push(
                  currentMessageRef.current.agentProgress.currentAgent
                );
                currentMessageRef.current.agentProgress.currentAgent = null;
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
          } else if (event.event === 'tool_call') {
            // Tool execution started
            console.log('🔧 FRONTEND: Received tool_call event:', event.data);
            const toolCall: ToolCall = {
              id: event.data.call_id || `tool_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              name: event.data.tool_name,
              input: event.data.tool_input,
              status: 'running',
            };
            console.log('🔧 FRONTEND: Created ToolCall:', toolCall);

            currentToolCallsRef.current.set(toolCall.id, toolCall);
            console.log('🔧 FRONTEND: currentToolCallsRef size:', currentToolCallsRef.current.size);

            if (currentMessageRef.current) {
              currentMessageRef.current.toolCalls = Array.from(
                currentToolCallsRef.current.values()
              );
              console.log('🔧 FRONTEND: Updated currentMessageRef.toolCalls:', currentMessageRef.current.toolCalls);
            } else {
              console.error('❌ FRONTEND: currentMessageRef.current is NULL!');
            }

            // Update UI
            setConversations((prev) => {
              console.log('🔧 FRONTEND: Updating conversations state...');
              return prev.map((c) => {
                if (c.id !== updatedConversation.id) return c;

                const messages = [...c.messages];
                const lastMsg = messages[messages.length - 1];
                console.log('🔧 FRONTEND: Last message role:', lastMsg?.role);

                if (lastMsg?.role === 'assistant') {
                  console.log('🔧 FRONTEND: Updating existing assistant message');
                  messages[messages.length - 1] = { ...currentMessageRef.current! };
                } else {
                  console.log('🔧 FRONTEND: Adding new assistant message');
                  messages.push({ ...currentMessageRef.current! });
                }

                console.log('🔧 FRONTEND: Updated message toolCalls:', messages[messages.length - 1]?.toolCalls);
                return { ...c, messages };
              });
            });
          } else if (event.event === 'tool_result') {
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
          } else if (event.event === 'cost_update') {
            // Cost tracking update
            if (currentMessageRef.current) {
              currentMessageRef.current.cost = {
                totalCost: event.data.total_cost,
                inputTokens: event.data.input_tokens,
                outputTokens: event.data.output_tokens,
                cachedTokens: event.data.cached_tokens,
                summary: event.data.summary,
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
          } else if (event.event === 'done') {
            // Stream completed
            break;
          } else if (event.event === 'error') {
            // Error occurred
            console.error('Stream error:', event.data);

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

              const finalConv = { ...c, messages: cleanedMessages, updatedAt: new Date() };

              // Save to localStorage with final content
              storageService.saveConversation(finalConv);

              return finalConv;
            })
          );
        }
      } catch (error) {
        console.error('❌ Error during streaming:', error);
        console.error('Error stack:', (error as Error).stack);

        // Add error message
        if (currentMessageRef.current) {
          currentMessageRef.current.content += `\n\n[Error: ${(error as Error).message}]`;
        }
      } finally {
        setIsStreaming(false);
        currentMessageRef.current = null;
        currentToolCallsRef.current = new Map();
      }
    },
    [isStreaming, createConversation, selectedModel, currentConversationId, conversations]
  );

  /**
   * Switch model
   */
  const switchModel = useCallback(async (model: string) => {
    try {
      await apiService.switchModel(model);
      setSelectedModel(model);
      storageService.setSelectedModel(model);
    } catch (error) {
      console.error('Failed to switch model:', error);
      throw error;
    }
  }, []);

  /**
   * Edit a user message and resend
   */
  const editMessage = useCallback(
    async (messageId: string, newContent: string) => {
      if (isStreaming || !newContent.trim()) return;

      // Use functional update to get current conversation state
      let shouldSend = false;

      setConversations((prev) => {
        // Find the current conversation
        const currentConv = prev.find((c) => c.id === currentConversationId);
        if (!currentConv) return prev;

        // Find the message index
        const messageIndex = currentConv.messages.findIndex((m) => m.id === messageId);
        if (messageIndex === -1 || currentConv.messages[messageIndex].role !== 'user') {
          return prev;
        }

        shouldSend = true;

        // Remove all messages after this one (including assistant response)
        const updatedMessages = currentConv.messages.slice(0, messageIndex);

        // Clean messages before saving
        const cleanedMessages = cleanMessages(updatedMessages);

        // Update conversation with truncated messages
        const updatedConversation: Conversation = {
          ...currentConv,
          messages: cleanedMessages,
          updatedAt: new Date(),
        };

        // Save to storage immediately
        storageService.saveConversation(updatedConversation);

        // Return updated conversations array
        return prev.map((c) => (c.id === currentConversationId ? updatedConversation : c));
      });

      // If validation passed, send the edited message
      if (shouldSend) {
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
        updatedAt: new Date(),
      };

      // Update state
      setConversations((prev) =>
        prev.map((c) => (c.id === conversationId ? updatedConversation : c))
      );

      // Save to storage
      storageService.saveConversation(updatedConversation);

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
    [isStreaming, sendMessage, currentConversationId]
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
        timestamp: new Date(),
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
          } else if (event.event === 'cost_update') {
            // Cost tracking update
            if (currentMessageRef.current) {
              currentMessageRef.current.cost = {
                totalCost: event.data.total_cost,
                inputTokens: event.data.input_tokens,
                outputTokens: event.data.output_tokens,
                cachedTokens: event.data.cached_tokens,
                summary: event.data.summary,
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
            // Stream completed
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
              const finalConv = { ...c, messages: cleanedMessages, updatedAt: new Date() };

              storageService.saveConversation(finalConv);

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

  return {
    conversations,
    currentConversation,
    isStreaming,
    selectedModel,
    clarificationData,
    awaitingClarification,
    createConversation,
    selectConversation,
    deleteConversation,
    sendMessage,
    switchModel,
    editMessage,
    regenerateMessage,
    submitClarification,
    cancelClarification,
  };
}
