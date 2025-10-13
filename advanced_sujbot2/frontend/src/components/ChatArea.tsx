import React, { useEffect, useRef, useState } from 'react';
import { ArrowDown } from 'lucide-react';
import { useChatStore } from '@/stores/chatStore';
import { BotMessage } from './BotMessage';
import { UserMessage } from './UserMessage';
import { ChatInput } from './ChatInput';
import { Button } from './ui/button';

export const ChatArea: React.FC = () => {
  const { messages, sendMessage } = useChatStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const userScrolledRef = useRef(false);
  const lastScrollTopRef = useRef(0);
  const isAutoScrollingRef = useRef(false);
  const [showScrollButton, setShowScrollButton] = useState(false);

  const scrollToBottom = (force = false) => {
    // Only auto-scroll if user hasn't manually scrolled up, or if forced
    if (force || !userScrolledRef.current) {
      const container = scrollContainerRef.current;
      if (container) {
        isAutoScrollingRef.current = true;
        container.scrollTo({
          top: container.scrollHeight,
          behavior: 'smooth'
        });
        // Reset flag after animation completes
        setTimeout(() => {
          isAutoScrollingRef.current = false;
        }, 500);
      }
    }
  };

  const handleScroll = () => {
    // Ignore scroll events triggered by auto-scrolling
    if (isAutoScrollingRef.current) return;

    const container = scrollContainerRef.current;
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 100;

    // Detect user-initiated upward scroll
    const scrollingUp = scrollTop < lastScrollTopRef.current;
    lastScrollTopRef.current = scrollTop;

    if (scrollingUp && !isAtBottom) {
      // User is actively scrolling up - disable auto-scroll
      userScrolledRef.current = true;
      setShowScrollButton(true);
    } else if (isAtBottom) {
      // User scrolled back to bottom - re-enable auto-scroll
      userScrolledRef.current = false;
      setShowScrollButton(false);
    }
  };

  const handleScrollToBottomClick = () => {
    userScrolledRef.current = false;
    setShowScrollButton(false);
    scrollToBottom(true);
  };

  useEffect(() => {
    // Auto-scroll for new messages and during streaming
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      // Auto-scroll if the last message is streaming or just added
      if (lastMessage.isStreaming || !userScrolledRef.current) {
        scrollToBottom();
      }
    }
  }, [messages]); // Trigger on any message changes (new messages + streaming updates)

  return (
    <div className="h-full flex flex-col relative overflow-hidden bg-gradient-to-br from-white via-gray-50 to-white dark:from-black dark:via-gray-950 dark:to-black transition-all duration-500">
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto overflow-x-hidden p-4 space-y-4 scroll-smooth"
        style={{
          scrollBehavior: 'smooth',
          overscrollBehavior: 'contain',
          WebkitOverflowScrolling: 'touch'
        }}
      >
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

      {/* Scroll to bottom button - positioned above ChatInput */}
      {showScrollButton && (
        <div className="absolute bottom-[100px] right-8 z-10 animate-fade-in">
          <Button
            onClick={handleScrollToBottomClick}
            size="icon"
            className="rounded-full shadow-lg bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 border border-gray-200 dark:border-gray-700 transition-all hover:scale-110"
            aria-label="Scroll to bottom"
          >
            <ArrowDown className="h-4 w-4 text-gray-700 dark:text-gray-300" />
          </Button>
        </div>
      )}

      <ChatInput onSend={sendMessage} />
    </div>
  );
};
