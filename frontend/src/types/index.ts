/**
 * TypeScript types for frontend application
 */

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  toolCalls?: ToolCall[];
  cost?: CostInfo;
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

export interface CostInfo {
  totalCost: number;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  summary: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export interface Model {
  id: string;
  name: string;
  provider: 'anthropic' | 'openai' | 'google';
  description: string;
}

export interface SSEEvent {
  event: 'text_delta' | 'tool_call' | 'tool_result' | 'tool_calls_summary' | 'cost_update' | 'done' | 'error' | 'clarification_needed';
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
