import React, { useState } from 'react';
import { User, Edit2, Check, X } from 'lucide-react';
import { ChatMessage } from '@/types';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { formatTimestamp } from '@/utils/date';
import { useChatStore } from '@/stores/chatStore';

interface UserMessageProps {
  message: ChatMessage;
}

export const UserMessage: React.FC<UserMessageProps> = ({ message }) => {
  const { editMessage, editingMessageId, setEditingMessage } = useChatStore();
  const [editedText, setEditedText] = useState(message.content);

  const isEditing = editingMessageId === message.id;

  const handleStartEdit = () => {
    setEditingMessage(message.id);
    setEditedText(message.content);
  };

  const handleCancelEdit = () => {
    setEditingMessage(null);
    setEditedText(message.content);
  };

  const handleSaveEdit = async () => {
    if (editedText.trim() && editedText !== message.content) {
      await editMessage(message.id, editedText);
    } else {
      handleCancelEdit();
    }
  };

  return (
    <div className="group flex gap-3 p-4 hover:bg-gray-50 dark:hover:bg-gray-900 rounded-lg transition-colors">
      <Avatar>
        <AvatarFallback className="bg-blue-500 text-white">
          <User className="h-5 w-5" />
        </AvatarFallback>
      </Avatar>

      <div className="flex-1 min-w-0">
        {isEditing ? (
          <div className="space-y-2">
            <Textarea
              value={editedText}
              onChange={(e) => setEditedText(e.target.value)}
              className="min-h-[80px] resize-none"
              autoFocus
            />
            <div className="flex gap-2">
              <Button
                size="sm"
                onClick={handleSaveEdit}
                disabled={!editedText.trim()}
              >
                <Check className="h-4 w-4 mr-1" />
                Save & Submit
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={handleCancelEdit}
              >
                <X className="h-4 w-4 mr-1" />
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <>
            <p className="text-sm text-gray-900 dark:text-gray-100 whitespace-pre-wrap">
              {message.content}
            </p>

            <div className="flex items-center gap-2 mt-2">
              <p className="text-xs text-gray-500">
                {formatTimestamp(message.timestamp)}
              </p>
              <Button
                size="sm"
                variant="ghost"
                className="opacity-0 group-hover:opacity-100 transition-opacity h-6 px-2"
                onClick={handleStartEdit}
              >
                <Edit2 className="h-3 w-3 mr-1" />
                Edit
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
