/**
 * API Service for backend communication
 *
 * Handles:
 * - SSE streaming for chat
 * - Authentication (cookie-based with httpOnly)
 * - Model switching
 * - Health checks
 */

import type { Model, HealthStatus, SSEEvent, Conversation, Message } from '../types';

// Use environment variable for API base URL
// Empty string = relative URLs (same-origin, through Nginx proxy)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL !== undefined
  ? import.meta.env.VITE_API_BASE_URL
  : 'http://localhost:8000';

// Authentication types
export interface UserProfile {
  id: number;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_admin: boolean;
  created_at: string;
  last_login_at: string | null;
}

export interface LoginResponse {
  user: UserProfile;
  message: string;
}

export class ApiService {
  /**
   * Get headers for JSON requests
   * (Cookies are sent automatically with credentials: 'include')
   */
  private getHeaders(): HeadersInit {
    return {
      'Content-Type': 'application/json',
    };
  }

  /**
   * Login with email and password
   * Backend sets httpOnly cookie with JWT token
   */
  async login(email: string, password: string): Promise<LoginResponse> {
    // Add timeout to prevent infinite loading state
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout for login

    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: this.getHeaders(),
        credentials: 'include', // Send/receive cookies
        body: JSON.stringify({ email, password }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Login failed' }));
        throw new Error(error.detail || 'Invalid credentials');
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Logout (clears httpOnly cookie on backend)
   */
  async logout(): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/auth/logout`, {
      method: 'POST',
      headers: this.getHeaders(),
      credentials: 'include', // Send cookies to backend
    });

    if (!response.ok) {
      throw new Error('Logout failed');
    }
  }

  /**
   * Get current user profile (validates JWT cookie)
   */
  async getCurrentUser(): Promise<UserProfile> {
    // Add timeout to prevent infinite loading state if backend is unreachable
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

    try {
      const response = await fetch(`${API_BASE_URL}/auth/me`, {
        method: 'GET',
        headers: this.getHeaders(),
        credentials: 'include', // Send cookies for authentication
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error('Authentication failed');
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Stream chat response using Server-Sent Events (SSE)
   */
  async *streamChat(
    message: string,
    conversationId?: string,
    skipSaveUserMessage?: boolean
  ): AsyncGenerator<SSEEvent, void, unknown> {
    let response;
    try {
      response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: this.getHeaders(),
        credentials: 'include', // Send authentication cookie
        body: JSON.stringify({
          message,
          conversation_id: conversationId,
          skip_save_user_message: skipSaveUserMessage || false,
        }),
      });
    } catch (error) {
      console.error('❌ API: Fetch failed:', error);
      // Yield error event to surface network failure to UI
      yield {
        event: 'error',
        data: {
          error: `Failed to connect to backend: ${(error as Error).message}. Check if backend is running on ${API_BASE_URL}`,
          type: 'NetworkError'
        }
      };
      return;  // Stop generator, don't throw (error already surfaced)
    }

    if (!response.ok) {
      // Yield error event for HTTP errors (4xx, 5xx)
      yield {
        event: 'error',
        data: {
          error: `Backend returned HTTP ${response.status}. This may indicate a server error.`,
          type: 'HTTPError',
          status: response.status
        }
      };
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      // Yield error event instead of throwing (consistent with other error handling)
      yield {
        event: 'error',
        data: {
          error: 'Backend response has no body. This may indicate a server configuration issue.',
          type: 'NoResponseBody'
        }
      };
      return;
    }

    const decoder = new TextDecoder();
    let buffer = '';

    // Timeout to prevent hanging on backend issues (10 minutes)
    const STREAM_TIMEOUT_MS = 10 * 60 * 1000;
    let timedOut = false;
    const timeoutId = setTimeout(() => {
      timedOut = true;
      reader.cancel('Stream timeout after 10 minutes');
      console.error('⏰ API: Stream timeout - cancelling reader');
    }, STREAM_TIMEOUT_MS);

    try {
      while (true) {
        const { done, value } = await reader.read();

        // Check for timeout (cancels reader, so next read may be done or throw)
        if (timedOut) {
          yield {
            event: 'error',
            data: {
              error: 'Stream timeout after 10 minutes. The backend may be experiencing performance issues or the query is too complex. Try a simpler question or refresh the page.',
              type: 'TimeoutError'
            }
          };
          break;
        }

        if (done) {
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // Process complete SSE messages
        // Note: Backend sends CRLF (\r\n\r\n) so we need to split on that
        const lines = buffer.split(/\r?\n\r?\n/);

        // If split returned only 1 element, buffer doesn't contain separator yet
        // Keep waiting for more data
        if (lines.length === 1) {
          continue;
        }

        // Last element is either incomplete or empty (if buffer ends with separator)
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          // Skip SSE comments (ping/keepalive) - they start with ":"
          // Example: ": ping - 2025-11-04 10:14:29.063439+00:00"
          if (line.trim().startsWith(':')) {
            continue;
          }

          // Parse SSE format:
          // event: text_delta
          // data: {"content": "Hello"}
          const eventMatch = line.match(/^event:\s*(.+)$/m);
          const dataMatch = line.match(/^data:\s*(.+)$/m);

          if (eventMatch && dataMatch) {
            const event = eventMatch[1].trim();

            // Try to parse JSON data
            try {
              const data = JSON.parse(dataMatch[1]);

              yield {
                event: event as SSEEvent['event'],
                data,
              };
            } catch (parseError) {
              // JSON parsing failed - this is a serious error, not just a warning
              console.error('❌ API: Failed to parse JSON in SSE data field:', {
                event,
                rawData: dataMatch[1],
                error: parseError
              });

              // Yield error event to surface the issue to UI
              yield {
                event: 'error',
                data: {
                  error: `Failed to parse server response (event: ${event})`,
                  type: 'JSONParseError',
                  rawData: dataMatch[1].substring(0, 100)
                }
              };
            }
          } else {
            // SSE format is invalid - this should NOT be just a warning
            console.error('❌ API: Invalid SSE format - missing event or data field:', {
              line,
              hasEvent: !!eventMatch,
              hasData: !!dataMatch
            });

            // Yield error event instead of silently dropping
            yield {
              event: 'error',
              data: {
                error: 'Server sent malformed response',
                type: 'SSEFormatError',
                details: line.substring(0, 100)
              }
            };
          }
        }
      }
    } finally {
      clearTimeout(timeoutId);
      reader.releaseLock();
    }
  }

  /**
   * Get list of available models
   */
  async getModels(): Promise<{ models: Model[]; defaultModel: string }> {
    const response = await fetch(`${API_BASE_URL}/models`, {
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Switch to a different model
   */
  async switchModel(model: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/model/switch?model=${encodeURIComponent(model)}`, {
      method: 'POST',
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to switch model: ${response.status}`);
    }
  }

  /**
   * Check backend health
   */
  async checkHealth(): Promise<HealthStatus> {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Stream clarification response using Server-Sent Events (SSE)
   *
   * Called when user responds to clarification questions. Resumes the workflow.
   */
  async *streamClarification(
    threadId: string,
    userResponse: string
  ): AsyncGenerator<SSEEvent, void, unknown> {
    let response;
    try {
      response = await fetch(`${API_BASE_URL}/chat/clarify`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({
          thread_id: threadId,
          response: userResponse,
        }),
      });
    } catch (error) {
      console.error('❌ API: Clarification fetch failed:', error);
      yield {
        event: 'error',
        data: {
          error: `Failed to connect to backend: ${(error as Error).message}`,
          type: 'NetworkError'
        }
      };
      return;
    }

    if (!response.ok) {
      yield {
        event: 'error',
        data: {
          error: `Backend returned HTTP ${response.status}`,
          type: 'HTTPError',
          status: response.status
        }
      };
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      yield {
        event: 'error',
        data: {
          error: 'Backend response has no body',
          type: 'NoResponseBody'
        }
      };
      return;
    }

    const decoder = new TextDecoder();
    let buffer = '';

    // Timeout (10 minutes)
    const STREAM_TIMEOUT_MS = 10 * 60 * 1000;
    let timedOut = false;
    const timeoutId = setTimeout(() => {
      timedOut = true;
      reader.cancel('Stream timeout after 10 minutes');
    }, STREAM_TIMEOUT_MS);

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (timedOut) {
          yield {
            event: 'error',
            data: {
              error: 'Stream timeout after 10 minutes',
              type: 'TimeoutError'
            }
          };
          break;
        }

        if (done) {
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const lines = buffer.split(/\r?\n\r?\n/);

        if (lines.length === 1) {
          continue;
        }

        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim() || line.trim().startsWith(':')) {
            continue;
          }

          const eventMatch = line.match(/^event:\s*(.+)$/m);
          const dataMatch = line.match(/^data:\s*(.+)$/m);

          if (eventMatch && dataMatch) {
            const event = eventMatch[1].trim();

            try {
              const data = JSON.parse(dataMatch[1]);
              yield {
                event: event as SSEEvent['event'],
                data,
              };
            } catch (parseError) {
              console.error('❌ API: Failed to parse clarification JSON:', parseError);
              yield {
                event: 'error',
                data: {
                  error: `Failed to parse server response (event: ${event})`,
                  type: 'JSONParseError'
                }
              };
            }
          }
        }
      }
    } finally {
      clearTimeout(timeoutId);
      reader.releaseLock();
    }
  }

  /**
   * Delete a message from conversation history
   */
  async deleteMessage(conversationId: string, messageId: string): Promise<void> {
    const response = await fetch(
      `${API_BASE_URL}/chat/${encodeURIComponent(conversationId)}/messages/${encodeURIComponent(messageId)}`,
      {
        method: 'DELETE',
        headers: this.getHeaders(),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to delete message: ${response.status}`);
    }
  }

  // ============================================================================
  // Conversation Management
  // ============================================================================

  /**
   * Get all conversations for the authenticated user
   */
  async getConversations(): Promise<Conversation[]> {
    const response = await fetch(`${API_BASE_URL}/conversations`, {
      method: 'GET',
      headers: this.getHeaders(),
      credentials: 'include', // Send JWT cookie
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch conversations: ${response.status}`);
    }

    const conversations = await response.json();

    // Convert backend snake_case to frontend camelCase
    return conversations.map((conv: any) => ({
      id: conv.id,
      title: conv.title,
      messages: conv.messages || [],  // Backend now includes empty messages array
      createdAt: conv.created_at,
      updatedAt: conv.updated_at,
      userId: conv.user_id,
    }));
  }

  /**
   * Create a new conversation
   */
  async createConversation(title?: string): Promise<Conversation> {
    const response = await fetch(`${API_BASE_URL}/conversations`, {
      method: 'POST',
      headers: this.getHeaders(),
      credentials: 'include',
      body: JSON.stringify({ title: title || 'New Conversation' }),
    });

    if (!response.ok) {
      throw new Error(`Failed to create conversation: ${response.status}`);
    }

    const data = await response.json();

    // Convert backend snake_case to frontend camelCase
    return {
      id: data.id,
      title: data.title,
      messages: data.messages || [],  // Backend now includes empty messages array
      createdAt: data.created_at,
      updatedAt: data.updated_at,
      userId: data.user_id,
    };
  }

  /**
   * Get message history for a conversation
   */
  async getConversationHistory(conversationId: string): Promise<Message[]> {
    const response = await fetch(
      `${API_BASE_URL}/conversations/${encodeURIComponent(conversationId)}/messages`,
      {
        method: 'GET',
        headers: this.getHeaders(),
        credentials: 'include',
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch conversation history: ${response.status}`);
    }

    const messages = await response.json();

    // Map backend format to frontend format
    return messages.map((msg: any) => {
      // Transform cost data from snake_case (backend) to camelCase (frontend)
      let cost = undefined;
      if (msg.metadata?.cost) {
        const backendCost = msg.metadata.cost;
        cost = {
          totalCost: backendCost.total_cost ?? 0,
          inputTokens: backendCost.total_input_tokens ?? 0,
          outputTokens: backendCost.total_output_tokens ?? 0,
          cachedTokens: backendCost.cache_stats?.cache_read_tokens ?? 0,
          agentBreakdown: backendCost.agent_breakdown ?? [],
          cacheStats: backendCost.cache_stats ?? {
            cache_read_tokens: 0,
            cache_creation_tokens: 0
          }
        };
      }

      return {
        id: msg.id,
        role: msg.role,
        content: msg.content,
        timestamp: msg.created_at, // Map created_at to timestamp
        metadata: msg.metadata,
        cost, // Transformed cost data
        toolCalls: msg.metadata?.tool_calls, // Map toolCalls from metadata if present
      };
    });
  }

  /**
   * Delete a conversation
   */
  async deleteConversation(conversationId: string): Promise<void> {
    const response = await fetch(
      `${API_BASE_URL}/conversations/${encodeURIComponent(conversationId)}`,
      {
        method: 'DELETE',
        headers: this.getHeaders(),
        credentials: 'include',
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to delete conversation: ${response.status}`);
    }
  }

  /**
   * Update conversation title
   */
  async updateConversationTitle(conversationId: string, title: string): Promise<void> {
    const response = await fetch(
      `${API_BASE_URL}/conversations/${encodeURIComponent(conversationId)}/title`,
      {
        method: 'PATCH',
        headers: this.getHeaders(),
        credentials: 'include',
        body: JSON.stringify({ title }),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to update conversation title: ${response.status}`);
    }
  }

  /**
   * Truncate messages after a certain count
   * Used for edit/regenerate to remove old messages
   */
  async truncateMessagesAfter(conversationId: string, keepCount: number): Promise<void> {
    const response = await fetch(
      `${API_BASE_URL}/conversations/${encodeURIComponent(conversationId)}/messages/after/${keepCount}`,
      {
        method: 'DELETE',
        headers: this.getHeaders(),
        credentials: 'include',
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to truncate messages: ${response.status}`);
    }
  }
}

// Singleton instance
export const apiService = new ApiService();
