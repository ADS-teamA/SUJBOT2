/**
 * TypeScript types for frontend application
 */

export interface AgentProgress {
  currentAgent: string | null;
  currentMessage: string | null;
  completedAgents: string[];
  activeTools: Array<{
    tool: string;
    status: 'running' | 'completed' | 'failed';
    timestamp: string;
  }>;
  /** Whether response is still being streamed (stays true until done event) */
  isStreaming?: boolean;
}

/**
 * Document information for PDF browser
 */
export interface DocumentInfo {
  document_id: string;
  display_name: string;
  filename: string;
  size_bytes: number;
}

export interface ToolHealth {
  healthy: boolean;
  summary: string;
  unavailableTools?: Record<string, string>;  // tool_name -> reason
  degradedTools?: string[];
}

/**
 * Context selected from PDF when user message was sent.
 * Stored in user messages to show indicator below message bubble.
 */
export interface MessageSelectedContext {
  /** Document ID for potential future features (e.g., click to open PDF) */
  documentId?: string;
  /** Human-readable document name */
  documentName: string;
  /** Number of non-empty lines in selection */
  lineCount: number;
  /** Starting page number (1-indexed) */
  pageStart: number;
  /** Ending page number (1-indexed) */
  pageEnd: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;  // ISO 8601 format (NOT Date object) for JSON serialization safety
  toolCalls?: ToolCall[];
  cost?: CostInfo;
  agentProgress?: AgentProgress;
  toolHealth?: ToolHealth;  // Tool availability status at query time
  selectedContext?: MessageSelectedContext;  // Context from PDF selection (user messages only)
}

export interface ToolCall {
  id: string;
  name: string;
  input: Record<string, any>;
  result?: any;
  executionTimeMs?: number;
  success?: boolean;
  status: 'running' | 'completed' | 'failed';
  explicitParams?: string[];
}

export interface AgentCostBreakdown {
  agent: string;                    // Agent name (e.g., "orchestrator", "extractor")
  cost: number;                     // Total cost in USD
  input_tokens: number;             // Total input tokens consumed
  output_tokens: number;            // Total output tokens generated
  cache_read_tokens: number;        // Tokens read from cache (Anthropic only)
  cache_creation_tokens: number;    // Tokens written to cache (Anthropic only)
  call_count: number;               // Number of LLM calls made by this agent
  response_time_ms: number;         // Total accumulated LLM response time in milliseconds
}

export interface CostInfo {
  totalCost: number;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  agentBreakdown?: AgentCostBreakdown[];
  cacheStats?: {
    cache_read_tokens: number;
    cache_creation_tokens: number;
  };
}

export interface Conversation {
  id: string;
  title: string;
  messageCount: number;  // Real message count from database (not messages.length which is lazy-loaded)
  messages: Message[];
  createdAt: string;  // ISO 8601 format for JSON serialization safety
  updatedAt: string;  // ISO 8601 format for JSON serialization safety
  userId?: number;  // Optional: included in responses from backend (user_id field)
}

export interface SSEEvent {
  event: 'tool_health' | 'text_delta' | 'tool_call' | 'tool_result' | 'tool_calls_summary' | 'cost_summary' | 'done' | 'error' | 'clarification_needed' | 'agent_start' | 'progress' | 'title_update';
  data: any;
}

export interface ClarificationQuestion {
  id: string;
  text: string;
  type: string;
}

export interface ClarificationData {
  thread_id: string;
  questions: ClarificationQuestion[];
  quality_metrics: {
    retrieval_score?: number;
    semantic_coherence?: number;
    query_pattern_score?: number;
    document_diversity?: number;
    overall_quality?: number;
  };
  original_query: string;
  complexity_score: number;
  timeout_seconds: number;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'error';
  message: string;
  details: Record<string, any>;
  degraded_components?: Array<{
    component: string;
    error: string;
  }>;
}

// ============================================================================
// Citation System Types
// ============================================================================

/**
 * Text selection from PDF for agent context.
 * Captures user-selected text to be included in the next query.
 */
export interface TextSelection {
  /** Selected text content */
  text: string;
  /** Source document ID */
  documentId: string;
  /** Human-readable document name */
  documentName: string;
  /** Starting page number (1-indexed) */
  pageStart: number;
  /** Ending page number (1-indexed) */
  pageEnd: number;
  /** Character count of selection */
  charCount: number;
}

export interface CitationMetadata {
  chunkId: string;
  documentId: string;
  documentName: string;
  sectionTitle: string | null;
  sectionPath: string | null;
  hierarchicalPath: string | null;
  pageNumber: number | null;
  pdfAvailable: boolean;
  content: string | null;  // Chunk text content for PDF highlighting
}

export interface CitationContextValue {
  /** Cache of fetched citation metadata */
  citationCache: Map<string, CitationMetadata>;
  /** Currently active PDF viewer state */
  activePdf: {
    documentId: string;
    documentName: string;
    page: number;
    chunkId?: string;
  } | null;
  /** Open PDF side panel */
  openPdf: (documentId: string, documentName: string, page?: number, chunkId?: string) => void;
  /** Close PDF side panel */
  closePdf: () => void;
  /** Fetch and cache metadata for chunk IDs */
  fetchCitationMetadata: (chunkIds: string[]) => Promise<void>;
  /** Set of chunk IDs currently being loaded (per-citation loading state) */
  loadingIds: Set<string>;
  /** Error message if fetch failed */
  error: string | null;
  /** Clear error state */
  clearError: () => void;
  /** Currently selected text from PDF for agent context */
  selectedText: TextSelection | null;
  /** Set selected text from PDF */
  setSelectedText: (selection: TextSelection | null) => void;
  /** Clear selected text */
  clearSelection: () => void;
}
