import React from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage, Source } from '@/types';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { formatTimestamp } from '@/utils/date';

interface BotMessageProps {
  message: ChatMessage;
  onCitationClick?: (source: Source) => void;
}

export const BotMessage: React.FC<BotMessageProps> = ({ message, onCitationClick }) => {
  const { t } = useTranslation();

  return (
    <div className="flex gap-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
      <Avatar>
        <AvatarFallback className="bg-primary text-white">
          <Bot className="h-5 w-5" />
        </AvatarFallback>
      </Avatar>

      <div className="flex-1 min-w-0">
        <ReactMarkdown
          className="prose dark:prose-invert max-w-none"
          remarkPlugins={[remarkGfm]}
        >
          {message.content}
        </ReactMarkdown>

        {message.sources && message.sources.length > 0 && (
          <div className="mt-4">
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              {t('sources')}:
            </p>
            <div className="flex flex-wrap gap-2">
              {message.sources.map((source, idx) => (
                <Badge
                  key={idx}
                  variant="secondary"
                  className="cursor-pointer"
                  onClick={() => onCitationClick?.(source)}
                >
                  {source.legal_reference}
                  {source.page && ` (p. ${source.page})`}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {message.isStreaming && (
          <div className="mt-2">
            <Loader2 className="animate-spin h-4 w-4 text-gray-400" />
          </div>
        )}

        <p className="text-xs text-gray-500 mt-2">
          {formatTimestamp(message.timestamp)}
        </p>
      </div>
    </div>
  );
};
