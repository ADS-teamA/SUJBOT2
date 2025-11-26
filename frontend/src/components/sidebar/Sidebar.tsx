/**
 * Sidebar Component - Conversation history and navigation
 */

import { useState, useEffect, useRef } from 'react';
import { Plus, MessageSquare, Trash2, Pencil } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { useSlideIn } from '../../design-system/animations/hooks/useSlideIn';
import { useHover } from '../../design-system/animations/hooks/useHover';
import type { Conversation } from '../../types';

interface ContextMenuState {
  x: number;
  y: number;
  conversationId: string;
}

interface SidebarProps {
  conversations: Conversation[];
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
  onRenameConversation: (id: string, newTitle: string) => void;
}

export function Sidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  onRenameConversation,
}: SidebarProps) {
  const { t } = useTranslation();
  const newChatHover = useHover({ scale: true });
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);

  // Close context menu when clicking outside
  useEffect(() => {
    if (!contextMenu) return;

    const handleClick = () => setContextMenu(null);
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setContextMenu(null);
    };

    document.addEventListener('click', handleClick);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('click', handleClick);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [contextMenu]);

  const handleContextMenu = (e: React.MouseEvent, conversationId: string) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY, conversationId });
  };

  const handleStartRename = (conversationId: string) => {
    setContextMenu(null);
    setEditingId(conversationId);
  };

  const handleRename = (id: string, newTitle: string) => {
    if (newTitle.trim()) {
      onRenameConversation(id, newTitle);
    }
    setEditingId(null);
  };

  return (
    <div className="w-full h-full flex flex-col">
      {/* New conversation button */}
      <div className={cn(
        'p-4',
        'border-b border-accent-200 dark:border-accent-800',
        'transition-colors duration-700'
      )}>
        <button
          onClick={onNewConversation}
          {...newChatHover.hoverProps}
          style={newChatHover.style}
          className={cn(
            'w-full flex items-center justify-center gap-2',
            'px-4 py-2 rounded-lg',
            'bg-accent-700 dark:bg-accent-300',
            'text-white dark:text-accent-900',
            'hover:bg-accent-800 dark:hover:bg-accent-400',
            'transition-all duration-700',
            'font-medium'
          )}
        >
          <Plus size={18} className="transition-all duration-700" />
          <span className="transition-colors duration-700">{t('sidebar.newChat')}</span>
        </button>
      </div>

      {/* Conversations list */}
      <div className="flex-1 overflow-y-auto p-2">
        {conversations.length === 0 ? (
          <div className={cn(
            'text-center py-8 text-sm',
            'text-accent-500 dark:text-accent-400',
            'transition-colors duration-700'
          )}>
            {t('sidebar.noConversations')}
          </div>
        ) : (
          <div className="space-y-1">
            {conversations
              .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())
              .map((conversation, index) => (
                <ConversationItem
                  key={conversation.id}
                  conversation={conversation}
                  isActive={currentConversationId === conversation.id}
                  isEditing={editingId === conversation.id}
                  onSelect={onSelectConversation}
                  onDelete={onDeleteConversation}
                  onContextMenu={handleContextMenu}
                  onRename={handleRename}
                  animationDelay={index * 30}
                />
              ))}
          </div>
        )}
      </div>

      {/* Context Menu */}
      {contextMenu && (
        <div
          className={cn(
            'fixed z-50 py-1 min-w-[140px]',
            'bg-white dark:bg-accent-900',
            'border border-accent-200 dark:border-accent-700',
            'rounded-lg shadow-lg',
            'transition-colors duration-700'
          )}
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={() => handleStartRename(contextMenu.conversationId)}
            className={cn(
              'w-full flex items-center gap-2 px-3 py-2',
              'text-sm text-left',
              'text-accent-700 dark:text-accent-300',
              'hover:bg-accent-100 dark:hover:bg-accent-800',
              'transition-colors duration-200'
            )}
          >
            <Pencil size={14} />
            {t('sidebar.rename')}
          </button>
          <button
            onClick={() => {
              onDeleteConversation(contextMenu.conversationId);
              setContextMenu(null);
            }}
            className={cn(
              'w-full flex items-center gap-2 px-3 py-2',
              'text-sm text-left',
              'text-red-600 dark:text-red-400',
              'hover:bg-red-50 dark:hover:bg-red-900/20',
              'transition-colors duration-200'
            )}
          >
            <Trash2 size={14} />
            {t('sidebar.delete')}
          </button>
        </div>
      )}
    </div>
  );
}

// Conversation item subcomponent with animations
interface ConversationItemProps {
  conversation: Conversation;
  isActive: boolean;
  isEditing: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onContextMenu: (e: React.MouseEvent, id: string) => void;
  onRename: (id: string, newTitle: string) => void;
  animationDelay: number;
}

function ConversationItem({
  conversation,
  isActive,
  isEditing,
  onSelect,
  onDelete,
  onContextMenu,
  onRename,
  animationDelay,
}: ConversationItemProps) {
  const { t } = useTranslation();
  const { style: slideStyle } = useSlideIn({
    direction: 'left',
    delay: animationDelay,
    duration: 'fast',
  });
  const deleteHover = useHover({ scale: true });
  const inputRef = useRef<HTMLInputElement>(null);
  const [editValue, setEditValue] = useState(conversation.title);

  // Focus input when entering edit mode
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
      setEditValue(conversation.title);
    }
  }, [isEditing, conversation.title]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      onRename(conversation.id, editValue);
    } else if (e.key === 'Escape') {
      setEditValue(conversation.title);
      onRename(conversation.id, conversation.title); // Cancel edit
    }
  };

  return (
    <div
      style={slideStyle}
      onClick={() => !isEditing && onSelect(conversation.id)}
      onContextMenu={(e) => onContextMenu(e, conversation.id)}
      className={cn(
        'group flex items-center gap-2 px-3 py-2 rounded-lg',
        'cursor-pointer transition-all duration-700',
        isActive
          ? 'bg-accent-200 dark:bg-accent-800 text-accent-900 dark:text-accent-100'
          : 'hover:bg-accent-100 dark:hover:bg-accent-800/50 hover:translate-x-1'
      )}
    >
      <MessageSquare size={16} className="flex-shrink-0 transition-all duration-700" />
      <div className="flex-1 min-w-0">
        {isEditing ? (
          <input
            ref={inputRef}
            type="text"
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={() => onRename(conversation.id, editValue)}
            onKeyDown={handleKeyDown}
            onClick={(e) => e.stopPropagation()}
            className={cn(
              'w-full px-1 py-0.5 text-sm font-medium',
              'bg-white dark:bg-accent-800',
              'border border-accent-300 dark:border-accent-600',
              'rounded outline-none',
              'focus:ring-2 focus:ring-accent-500 dark:focus:ring-accent-400',
              'transition-all duration-200'
            )}
          />
        ) : (
          <div className="text-sm font-medium truncate transition-colors duration-700">
            {conversation.title}
          </div>
        )}
        <div className={cn(
          'text-xs',
          'text-accent-500 dark:text-accent-400',
          'transition-colors duration-700'
        )}>
          {t('sidebar.messageCount', { count: conversation.messageCount })}
        </div>
      </div>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onDelete(conversation.id);
        }}
        {...deleteHover.hoverProps}
        style={deleteHover.style}
        className={cn(
          'opacity-0 group-hover:opacity-100',
          'p-1 rounded',
          'hover:bg-accent-300 dark:hover:bg-accent-700',
          'transition-all duration-700'
        )}
        title={t('sidebar.deleteConversation')}
      >
        <Trash2 size={14} className={cn(
          'text-accent-700 dark:text-accent-300',
          'transition-colors duration-700'
        )} />
      </button>
    </div>
  );
}
