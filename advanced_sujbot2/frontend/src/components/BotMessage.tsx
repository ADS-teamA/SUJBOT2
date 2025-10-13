import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, Bot, RefreshCw, FileText, ChevronDown, ChevronUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage, Source } from '@/types';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { formatTimestamp } from '@/utils/date';
import { useChatStore } from '@/stores/chatStore';

interface BotMessageProps {
  message: ChatMessage;
  onCitationClick?: (source: Source) => void;
}

export const BotMessage: React.FC<BotMessageProps> = ({ message, onCitationClick }) => {
  const { t } = useTranslation();
  const { regenerateResponse, isTyping } = useChatStore();
  const [isContextOpen, setIsContextOpen] = useState(false);

  const handleRegenerate = () => {
    regenerateResponse(message.id);
  };

  return (
    <div className="group flex gap-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
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

        {/* Collapsible Context (Sources) */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-4">
            <Button
              variant="outline"
              size="sm"
              className="w-full justify-between text-sm"
              onClick={() => setIsContextOpen(!isContextOpen)}
            >
              <span className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                {t('context')} ({message.sources.length} {message.sources.length === 1 ? t('source') : t('sources')})
              </span>
              {isContextOpen ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>

            {isContextOpen && (
              <div className="mt-2 p-3 bg-gray-100 dark:bg-gray-700 rounded-md border border-gray-200 dark:border-gray-600 animate-fade-in">
                <div className="flex flex-wrap gap-2">
                  {message.sources.map((source, idx) => (
                    <Badge
                      key={idx}
                      variant="secondary"
                      className="cursor-pointer hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                      onClick={() => onCitationClick?.(source)}
                    >
                      {source.legal_reference}
                      {source.page && ` (p. ${source.page})`}
                      {source.confidence && (
                        <span className="ml-1 text-xs opacity-70">
                          {(source.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Pipeline Status Display */}
        {message.isStreaming && (
          <div className="mt-3 flex items-center gap-3 text-sm text-gray-600 dark:text-gray-400">
            <Loader2 className="animate-spin h-4 w-4" />
            {message.pipelineStatus ? (
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium">{message.pipelineStatus.message}</span>
                  <span className="text-xs">
                    {message.pipelineStatus.step}/{message.pipelineStatus.total_steps}
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                  <div
                    className="bg-blue-600 dark:bg-blue-400 h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${message.pipelineStatus.progress}%` }}
                  />
                </div>
              </div>
            ) : (
              <span>{t('processing')}...</span>
            )}
          </div>
        )}

        <div className="flex items-center gap-2 mt-2">
          <p className="text-xs text-gray-500">
            {formatTimestamp(message.timestamp)}
          </p>
          {!message.isStreaming && (
            <Button
              size="sm"
              variant="ghost"
              className="opacity-0 group-hover:opacity-100 transition-opacity h-6 px-2"
              onClick={handleRegenerate}
              disabled={isTyping}
            >
              <RefreshCw className="h-3 w-3 mr-1" />
              Regenerate
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};
