/**
 * AdminConversationViewer - Read-only chat view for admin user inspection
 *
 * Features:
 * - Two-panel layout: conversation list + chat view
 * - Chat-style message display (same as user sees)
 * - Read-only (no edit/regenerate/delete)
 * - Markdown rendering with syntax highlighting
 */

import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { MessageCircle, Clock, User, Bot, Loader2, AlertCircle, ChevronRight } from 'lucide-react';
import { getApiBaseUrl } from '../dataProvider';
import './AdminConversationViewer.css';

interface Conversation {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

interface Message {
  id: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata?: Record<string, unknown>;
  created_at: string;
}

interface AdminConversationViewerProps {
  userId: number;
}

export function AdminConversationViewer({ userId }: AdminConversationViewerProps) {
  const { t } = useTranslation();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadingConversations, setLoadingConversations] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch conversations on mount
  useEffect(() => {
    fetchConversations();
  }, [userId]);

  // Fetch messages when conversation selected
  useEffect(() => {
    if (selectedConversation) {
      fetchMessages(selectedConversation);
    } else {
      setMessages([]);
    }
  }, [selectedConversation]);

  const fetchConversations = async () => {
    setLoadingConversations(true);
    setError(null);
    try {
      const response = await fetch(
        `${getApiBaseUrl()}/admin/users/${userId}/conversations`,
        {
          credentials: 'include',
          headers: { Accept: 'application/json' },
        }
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setConversations(data);
    } catch (e) {
      console.error('Failed to fetch conversations:', e);
      setError(t('admin.conversations.loadError'));
    } finally {
      setLoadingConversations(false);
    }
  };

  const fetchMessages = async (conversationId: string) => {
    setLoadingMessages(true);
    setError(null);
    try {
      const response = await fetch(
        `${getApiBaseUrl()}/admin/users/${userId}/conversations/${conversationId}/messages`,
        {
          credentials: 'include',
          headers: { Accept: 'application/json' },
        }
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setMessages(data);
    } catch (e) {
      console.error('Failed to fetch messages:', e);
      setError(t('admin.conversations.loadMessagesError'));
    } finally {
      setLoadingMessages(false);
    }
  };

  const formatDate = (isoString: string) => {
    return new Date(isoString).toLocaleString();
  };

  const selectedConversationData = conversations.find(c => c.id === selectedConversation);

  return (
    <div className="admin-conversation-viewer">
      {/* Error display */}
      {error && (
        <div className="acv-error">
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Conversation list panel */}
      <div className="acv-list-panel">
        <div className="acv-list-header">
          <MessageCircle size={18} />
          <span>{t('admin.conversations.title')}</span>
          <span className="acv-count">({conversations.length})</span>
        </div>

        {loadingConversations ? (
          <div className="acv-loading">
            <Loader2 size={20} className="acv-spinner" />
          </div>
        ) : conversations.length === 0 ? (
          <div className="acv-empty">
            {t('admin.conversations.noConversations')}
          </div>
        ) : (
          <ul className="acv-conversation-list">
            {conversations.map(conv => (
              <li
                key={conv.id}
                className={`acv-conversation-item ${selectedConversation === conv.id ? 'selected' : ''}`}
                onClick={() => setSelectedConversation(conv.id)}
              >
                <div className="acv-conv-main">
                  <span className="acv-conv-title">{conv.title}</span>
                  <ChevronRight size={16} className="acv-conv-arrow" />
                </div>
                <div className="acv-conv-meta">
                  <span className="acv-conv-count">
                    {conv.message_count} {t('admin.conversations.messages')}
                  </span>
                  <span className="acv-conv-date">
                    {formatDate(conv.updated_at)}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Chat view panel */}
      <div className="acv-chat-panel">
        {!selectedConversation ? (
          <div className="acv-no-selection">
            <MessageCircle size={48} className="acv-no-selection-icon" />
            <p>{t('admin.conversations.selectConversation')}</p>
          </div>
        ) : (
          <>
            <div className="acv-chat-header">
              <span className="acv-chat-title">
                {selectedConversationData?.title || 'Conversation'}
              </span>
              <span className="acv-read-only-badge">
                {t('admin.conversations.readOnly')}
              </span>
            </div>

            {loadingMessages ? (
              <div className="acv-loading">
                <Loader2 size={24} className="acv-spinner" />
              </div>
            ) : (
              <div className="acv-messages">
                {messages.map(msg => (
                  <div
                    key={msg.id}
                    className={`acv-message ${msg.role}`}
                  >
                    <div className="acv-message-header">
                      {msg.role === 'user' ? (
                        <User size={16} />
                      ) : msg.role === 'assistant' ? (
                        <Bot size={16} />
                      ) : (
                        <span className="acv-system-icon">S</span>
                      )}
                      <span className="acv-message-role">
                        {msg.role === 'user'
                          ? t('chat.you')
                          : msg.role === 'assistant'
                          ? t('chat.assistant')
                          : 'System'}
                      </span>
                      <span className="acv-message-time">
                        <Clock size={12} />
                        {formatDate(msg.created_at)}
                      </span>
                    </div>
                    <div className="acv-message-content">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
