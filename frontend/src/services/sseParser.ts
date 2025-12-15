/**
 * SSE Stream Parser Utility
 *
 * Parses Server-Sent Events from a ReadableStream.
 * Handles CRLF line endings, SSE comments (keepalive), and JSON data.
 *
 * Extracted from api.ts to eliminate code duplication between
 * streamChat() and streamClarification().
 */

import type { SSEEvent } from '../types';

/**
 * Configuration options for SSE parsing
 */
export interface SSEParserOptions {
  /** Timeout in milliseconds (default: 10 minutes) */
  timeoutMs?: number;
  /** Optional AbortSignal for cancellation */
  abortSignal?: AbortSignal;
  /** Callback when timeout occurs */
  onTimeout?: () => void;
}

/**
 * Parse SSE stream from a ReadableStreamDefaultReader
 *
 * @param reader - The stream reader from response.body.getReader()
 * @param options - Parser configuration options
 * @yields SSEEvent objects parsed from the stream
 *
 * @example
 * ```typescript
 * const reader = response.body?.getReader();
 * if (reader) {
 *   for await (const event of parseSSEStream(reader, { timeoutMs: 60000 })) {
 *     console.log(event.event, event.data);
 *   }
 * }
 * ```
 */
export async function* parseSSEStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  options: SSEParserOptions = {}
): AsyncGenerator<SSEEvent, void, unknown> {
  const { timeoutMs = 10 * 60 * 1000, abortSignal, onTimeout } = options;

  const decoder = new TextDecoder();
  let buffer = '';
  let timedOut = false;

  // Set up timeout
  const timeoutId = setTimeout(() => {
    timedOut = true;
    reader.cancel('Stream timeout');
    onTimeout?.();
    console.error('⏰ SSE Parser: Stream timeout - cancelling reader');
  }, timeoutMs);

  try {
    while (true) {
      // Check if abort was requested before reading
      if (abortSignal?.aborted) {
        console.log('📡 SSE Parser: Stream aborted before read');
        break;
      }

      let done: boolean;
      let value: Uint8Array | undefined;

      try {
        const result = await reader.read();
        done = result.done;
        value = result.value;
      } catch (readError) {
        // Check if read failed due to abort
        if (readError instanceof Error && readError.name === 'AbortError') {
          console.log('📡 SSE Parser: Stream read aborted');
          break;
        }
        throw readError;
      }

      // Check for timeout
      if (timedOut) {
        yield {
          event: 'error',
          data: {
            error: 'Stream timeout. The backend may be experiencing performance issues.',
            type: 'TimeoutError',
          },
        };
        break;
      }

      if (done) {
        break;
      }

      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;

      // Process complete SSE messages
      // Backend sends CRLF (\r\n\r\n) so we split on that
      const lines = buffer.split(/\r?\n\r?\n/);

      // If split returned only 1 element, buffer doesn't contain separator yet
      if (lines.length === 1) {
        continue;
      }

      // Last element is either incomplete or empty
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;

        // Skip SSE comments (ping/keepalive) - they start with ":"
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

          try {
            const data = JSON.parse(dataMatch[1]);
            yield {
              event: event as SSEEvent['event'],
              data,
            };
          } catch (parseError) {
            console.error('❌ SSE Parser: Failed to parse JSON:', {
              event,
              rawData: dataMatch[1],
              error: parseError,
            });

            yield {
              event: 'error',
              data: {
                error: `Failed to parse server response (event: ${event})`,
                type: 'JSONParseError',
                rawData: dataMatch[1].substring(0, 100),
              },
            };
          }
        } else {
          // SSE format is invalid
          console.error('❌ SSE Parser: Invalid SSE format:', {
            line,
            hasEvent: !!eventMatch,
            hasData: !!dataMatch,
          });

          yield {
            event: 'error',
            data: {
              error: 'Server sent malformed response',
              type: 'SSEFormatError',
              details: line.substring(0, 100),
            },
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
 * Create an error SSEEvent
 * Utility for consistent error event creation
 */
export function createErrorEvent(
  error: string,
  type: string,
  extra?: Record<string, unknown>
): SSEEvent {
  return {
    event: 'error',
    data: {
      error,
      type,
      ...extra,
    },
  };
}
