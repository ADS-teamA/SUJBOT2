import React from 'react';
import { User } from 'lucide-react';
import { ChatMessage } from '@/types';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { formatTimestamp } from '@/utils/date';

interface UserMessageProps {
  message: ChatMessage;
}

export const UserMessage: React.FC<UserMessageProps> = ({ message }) => {
  return (
    <div className="flex gap-3 p-4">
      <Avatar>
        <AvatarFallback className="bg-blue-500 text-white">
          <User className="h-5 w-5" />
        </AvatarFallback>
      </Avatar>

      <div className="flex-1 min-w-0">
        <p className="text-sm text-gray-900 dark:text-gray-100 whitespace-pre-wrap">
          {message.content}
        </p>

        <p className="text-xs text-gray-500 mt-2">
          {formatTimestamp(message.timestamp)}
        </p>
      </div>
    </div>
  );
};
