/**
 * Conversation API Client
 * Handles communication with backend conversation persistence endpoints
 */

import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export interface Message {
  id: string;
  conversation_id: string;
  type: 'user' | 'bot';
  content: string;
  created_at: string;
  sequence: number;
  intent?: string;
  pipeline?: string;
  sources?: any[];
  metadata?: Record<string, any>;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  is_archived: boolean;
  is_favorite: boolean;
  document_ids?: string[];
  session_id?: string;
  messages: Message[];
}

export interface ConversationListItem {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  is_archived: boolean;
  is_favorite: boolean;
  document_ids?: string[];
  latest_message_preview?: string;
  latest_message_type?: string;
}

export interface ConversationListResponse {
  conversations: ConversationListItem[];
  total: number;
  limit: number;
  offset: number;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Create a new conversation
 */
export async function createConversation(data: {
  title?: string;
  document_ids?: string[];
  session_id?: string;
}): Promise<Conversation> {
  const response = await axios.post(`${API_URL}/conversations`, data);
  return response.data;
}

/**
 * List conversations (paginated)
 */
export async function listConversations(params: {
  limit?: number;
  offset?: number;
  include_archived?: boolean;
} = {}): Promise<ConversationListResponse> {
  const response = await axios.get(`${API_URL}/conversations`, { params });
  return response.data;
}

/**
 * Get full conversation with messages
 */
export async function getConversation(conversationId: string): Promise<Conversation> {
  const response = await axios.get(`${API_URL}/conversations/${conversationId}`);
  return response.data;
}

/**
 * Update conversation metadata
 */
export async function updateConversation(
  conversationId: string,
  data: {
    title?: string;
    is_archived?: boolean;
    is_favorite?: boolean;
    document_ids?: string[];
  }
): Promise<Conversation> {
  const response = await axios.patch(`${API_URL}/conversations/${conversationId}`, data);
  return response.data;
}

/**
 * Delete conversation
 */
export async function deleteConversation(conversationId: string): Promise<void> {
  await axios.delete(`${API_URL}/conversations/${conversationId}`);
}

/**
 * Add message to conversation
 */
export async function addMessage(
  conversationId: string,
  message: {
    type: 'user' | 'bot';
    content: string;
    sequence: number;
    intent?: string;
    pipeline?: string;
    sources?: any[];
    metadata?: Record<string, any>;
  }
): Promise<Message> {
  const response = await axios.post(
    `${API_URL}/conversations/${conversationId}/messages`,
    message
  );
  return response.data;
}

/**
 * Add multiple messages in batch (for syncing)
 */
export async function addMessagesBatch(
  conversationId: string,
  messages: Array<{
    type: 'user' | 'bot';
    content: string;
    sequence: number;
    intent?: string;
    pipeline?: string;
    sources?: any[];
    metadata?: Record<string, any>;
  }>
): Promise<Message[]> {
  const response = await axios.post(
    `${API_URL}/conversations/${conversationId}/messages/batch`,
    { messages }
  );
  return response.data;
}
