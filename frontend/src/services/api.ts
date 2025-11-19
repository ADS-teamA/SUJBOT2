/**
 * API Service for backend communication
 *
 * Handles:
 * - SSE streaming for chat
 * - Model switching
 * - Health checks
 */

import type { Model, HealthStatus, SSEEvent } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export class ApiService {
  /**
   * Stream chat response using Server-Sent Events (SSE)
   */
  async *streamChat(
    message: string,
    conversationId?: string
  ): AsyncGenerator<SSEEvent, void, unknown> {
    let response;
    try {
      response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          conversation_id: conversationId,
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
    const response = await fetch(`${API_BASE_URL}/models`);

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
        headers: {
          'Content-Type': 'application/json',
        },
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
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to delete message: ${response.status}`);
    }
  }
}

// Singleton instance
export const apiService = new ApiService();
