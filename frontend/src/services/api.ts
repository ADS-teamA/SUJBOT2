/**
 * API Service for backend communication
 *
 * Handles:
 * - SSE streaming for chat (via sseParser utility)
 * - Authentication (cookie-based with httpOnly)
 * - Health checks
 * - Conversation management
 */

import type { HealthStatus, SSEEvent, Conversation, Message, AgentCostBreakdown } from '../types';
import { parseSSEStream } from './sseParser';

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

export interface SpendingInfo {
  total_spent_czk: number;
  spending_limit_czk: number;
  remaining_czk: number;
  reset_at: string | null;
}

export interface DocumentInfo {
  document_id: string;
  display_name: string;
  filename: string;
  size_bytes: number;
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
   * @param externalSignal - Optional AbortSignal for cancellation (e.g., component unmount)
   */
  async getCurrentUser(externalSignal?: AbortSignal): Promise<UserProfile> {
    // Add timeout to prevent infinite loading state if backend is unreachable
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

    // If external signal aborts, abort our controller too
    const abortHandler = () => controller.abort();
    externalSignal?.addEventListener('abort', abortHandler);

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
      externalSignal?.removeEventListener('abort', abortHandler);
    }
  }

  // Simple cache for spending data (prevents duplicate requests)
  private spendingCache: { data: SpendingInfo | null; timestamp: number } = {
    data: null,
    timestamp: 0,
  };
  private spendingCacheTTL = 2000; // 2 seconds cache

  /**
   * Get current user's spending information
   */
  async getSpending(): Promise<SpendingInfo> {
    // Return cached data if fresh (within TTL)
    const now = Date.now();
    if (this.spendingCache.data && now - this.spendingCache.timestamp < this.spendingCacheTTL) {
      return this.spendingCache.data;
    }

    const response = await fetch(`${API_BASE_URL}/settings/spending`, {
      method: 'GET',
      headers: this.getHeaders(),
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch spending: ${response.status}`);
    }

    const data = await response.json();

    // Update cache
    this.spendingCache = { data, timestamp: now };

    return data;
  }

  /**
   * Invalidate spending cache (call after message is sent)
   */
  invalidateSpendingCache(): void {
    this.spendingCache.timestamp = 0;
  }

  /**
   * Stream chat response using Server-Sent Events (SSE)
   * @param message - Current user message
   * @param conversationId - Optional conversation ID
   * @param skipSaveUserMessage - Skip saving user message (for regenerate)
   * @param messageHistory - Optional last N messages for conversation context
   * @param abortSignal - Optional AbortSignal for cancellation (e.g., on page refresh)
   * @param selectedContext - Optional selected text from PDF for additional context
   */
  async *streamChat(
    message: string,
    conversationId?: string,
    skipSaveUserMessage?: boolean,
    messageHistory?: Array<{ role: 'user' | 'assistant'; content: string }>,
    abortSignal?: AbortSignal,
    selectedContext?: {
      text: string;
      document_id: string;
      document_name: string;
      page_start: number;
      page_end: number;
    } | null
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
          messages: messageHistory,  // Conversation history for context
          selected_context: selectedContext || null,  // Selected text from PDF
        }),
        signal: abortSignal,  // Allow cancellation on page refresh/unmount
      });
    } catch (error) {
      // Check if this was an intentional abort (page refresh, navigation)
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('📡 API: Stream aborted by client (page refresh or navigation)');
        return;  // Clean exit, no error event needed
      }
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
      // Handle 402 Payment Required (spending limit exceeded) specially
      if (response.status === 402) {
        try {
          const errorData = await response.json();
          yield {
            event: 'error',
            data: {
              error: errorData.detail?.message_en || 'Spending limit exceeded',
              type: 'SpendingLimitExceeded',
              status: response.status,
              total_spent_czk: errorData.detail?.total_spent_czk,
              spending_limit_czk: errorData.detail?.spending_limit_czk,
              message_cs: errorData.detail?.message_cs,
              message_en: errorData.detail?.message_en,
            }
          };
        } catch (parseError) {
          console.error('Failed to parse 402 spending limit error response:', parseError);
          yield {
            event: 'error',
            data: {
              error: 'Spending limit exceeded. Contact administrator.',
              type: 'SpendingLimitExceeded',
              status: response.status
            }
          };
        }
        return;
      }

      // Yield error event for other HTTP errors (4xx, 5xx)
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
      yield {
        event: 'error',
        data: {
          error: 'Backend response has no body. This may indicate a server configuration issue.',
          type: 'NoResponseBody'
        }
      };
      return;
    }

    // Delegate SSE parsing to shared utility
    yield* parseSSEStream(reader, {
      timeoutMs: 10 * 60 * 1000,  // 10 minutes
      abortSignal,
    });
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
   * Get list of available documents
   */
  async getDocuments(): Promise<DocumentInfo[]> {
    const response = await fetch(`${API_BASE_URL}/documents/`, {
      method: 'GET',
      headers: this.getHeaders(),
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch documents: ${response.status}`);
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

    // Delegate SSE parsing to shared utility
    yield* parseSSEStream(reader, {
      timeoutMs: 10 * 60 * 1000,  // 10 minutes
    });
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
      messageCount: conv.message_count ?? 0,  // Real count from database
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
      messageCount: data.message_count ?? 0,  // New conversation starts with 0 messages
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
      // Transform top-level cost data from snake_case (backend) to camelCase (frontend)
      // Nested structures (agentBreakdown, cacheStats) preserve snake_case field names
      let cost = undefined;
      if (msg.metadata?.cost) {
        const backendCost = msg.metadata.cost;

        // Validate and transform agent breakdown with runtime checks
        let agentBreakdown: AgentCostBreakdown[] = [];
        if (backendCost.agent_breakdown) {
          if (!Array.isArray(backendCost.agent_breakdown)) {
            console.error(
              'Invalid agent_breakdown from backend - expected array, got:',
              typeof backendCost.agent_breakdown,
              backendCost.agent_breakdown
            );
          } else {
            agentBreakdown = backendCost.agent_breakdown.map((agent: any, idx: number) => {
              // Validate each agent entry
              if (typeof agent !== 'object' || agent === null) {
                console.warn(`Invalid agent entry at index ${idx}:`, agent);
                return null;
              }

              // Validate response_time_ms specifically (new field in this PR)
              if (agent.response_time_ms !== undefined && typeof agent.response_time_ms !== 'number') {
                console.warn(
                  `Invalid response_time_ms for agent ${agent.agent || 'unknown'}:`,
                  agent.response_time_ms
                );
              }

              return {
                agent: agent.agent ?? 'unknown',
                cost: typeof agent.cost === 'number' ? agent.cost : 0,
                input_tokens: typeof agent.input_tokens === 'number' ? agent.input_tokens : 0,
                output_tokens: typeof agent.output_tokens === 'number' ? agent.output_tokens : 0,
                cache_read_tokens: typeof agent.cache_read_tokens === 'number' ? agent.cache_read_tokens : 0,
                cache_creation_tokens: typeof agent.cache_creation_tokens === 'number' ? agent.cache_creation_tokens : 0,
                call_count: typeof agent.call_count === 'number' ? agent.call_count : 0,
                response_time_ms: typeof agent.response_time_ms === 'number' ? agent.response_time_ms : 0,
              };
            }).filter((agent: AgentCostBreakdown | null): agent is AgentCostBreakdown => agent !== null);
          }
        }

        cost = {
          totalCost: backendCost.total_cost ?? 0,
          inputTokens: backendCost.total_input_tokens ?? 0,
          outputTokens: backendCost.total_output_tokens ?? 0,
          cachedTokens: backendCost.cache_stats?.cache_read_tokens ?? 0,
          agentBreakdown,  // Validated array
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
