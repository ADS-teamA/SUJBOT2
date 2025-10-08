import React, { useEffect, useRef } from 'react';
import { useChatStore } from '@/stores/chatStore';
import { BotMessage } from './BotMessage';
import { UserMessage } from './UserMessage';
import { ChatInput } from './ChatInput';

export const ChatArea: React.FC = () => {
  const { messages, sendMessage } = useChatStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-white via-gray-50 to-white dark:from-black dark:via-gray-950 dark:to-black transition-all duration-500">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-500 dark:text-gray-400">
            <div className="text-center animate-fade-in-up">
              <div className="relative inline-block mb-4">
                <div className="absolute inset-0 bg-gradient-to-r from-gray-400 to-gray-600 dark:from-gray-400 dark:to-gray-200 rounded-full blur-xl opacity-20 animate-pulse-slow"></div>
                <h2 className="relative text-3xl font-bold bg-gradient-to-r from-black via-gray-700 to-black dark:from-white dark:via-gray-300 dark:to-white bg-clip-text text-transparent">
                  SUJBOT2
                </h2>
              </div>
              <p className="text-sm opacity-70">Legal Compliance Assistant</p>
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={message.id} className="animate-fade-in" style={{ animationDelay: `${index * 0.05}s` }}>
              {message.type === 'user' ? (
                <UserMessage message={message} />
              ) : (
                <BotMessage message={message} />
              )}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <ChatInput onSend={sendMessage} />
    </div>
  );
};
