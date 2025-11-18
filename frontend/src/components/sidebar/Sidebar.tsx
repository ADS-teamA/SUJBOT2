/**
 * Sidebar Component - Conversation history and navigation
 */

import { Plus, MessageSquare, Trash2 } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';
import { useSlideIn } from '../../design-system/animations/hooks/useSlideIn';
import { useHover } from '../../design-system/animations/hooks/useHover';
import type { Conversation } from '../../types';

interface SidebarProps {
  conversations: Conversation[];
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
}

export function Sidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
}: SidebarProps) {
  const newChatHover = useHover({ scale: true });

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
          <span className="transition-colors duration-700">New Chat</span>
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
            No conversations yet
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
                  onSelect={onSelectConversation}
                  onDelete={onDeleteConversation}
                  animationDelay={index * 30}
                />
              ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Conversation item subcomponent with animations
interface ConversationItemProps {
  conversation: Conversation;
  isActive: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  animationDelay: number;
}

function ConversationItem({
  conversation,
  isActive,
  onSelect,
  onDelete,
  animationDelay,
}: ConversationItemProps) {
  const { style: slideStyle } = useSlideIn({
    direction: 'left',
    delay: animationDelay,
    duration: 'fast',
  });
  const deleteHover = useHover({ scale: true });

  return (
    <div
      style={slideStyle}
      onClick={() => onSelect(conversation.id)}
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
        <div className="text-sm font-medium truncate transition-colors duration-700">
          {conversation.title}
        </div>
        <div className={cn(
          'text-xs',
          'text-accent-500 dark:text-accent-400',
          'transition-colors duration-700'
        )}>
          {conversation.messages.length} messages
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
        title="Delete conversation"
      >
        <Trash2 size={14} className={cn(
          'text-accent-700 dark:text-accent-300',
          'transition-colors duration-700'
        )} />
      </button>
    </div>
  );
}
