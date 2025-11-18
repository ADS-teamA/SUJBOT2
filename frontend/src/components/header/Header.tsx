/**
 * Header Component - Top navigation with theme toggle and sidebar control
 */

import { Sun, Moon, Menu } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';
import { useHover } from '../../design-system/animations/hooks/useHover';

interface HeaderProps {
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
  onToggleSidebar: () => void;
  sidebarOpen: boolean;
}

export function Header({
  theme,
  onToggleTheme,
  onToggleSidebar,
  sidebarOpen,
}: HeaderProps) {
  // Animation hooks
  const hamburgerHover = useHover({ scale: true });
  const themeHover = useHover({ scale: true });

  return (
    <header className={cn(
      'bg-white dark:bg-accent-900',
      'border-b border-accent-200 dark:border-accent-800',
      'px-6 py-4',
      'transition-all duration-700'
    )}>
      <div className="flex items-center justify-between">
        {/* Left side: Hamburger + Logo */}
        <div className="flex items-center gap-3">
          {/* Hamburger button */}
          <button
            onClick={onToggleSidebar}
            {...hamburgerHover.hoverProps}
            style={hamburgerHover.style}
            className={cn(
              'p-2 rounded-lg',
              'text-accent-700 dark:text-accent-300',
              'hover:bg-accent-100 dark:hover:bg-accent-800',
              'transition-all duration-700'
            )}
            aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          >
            <Menu size={20} className="transition-all duration-700" />
          </button>

          {/* Logo and title */}
          <div className="flex items-center gap-3">
            {/* Icon - Inline SVG for proper currentColor support */}
            <svg
              width="40"
              height="40"
              viewBox="0 0 64 64"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className={cn(
                'text-accent-900 dark:text-accent-100',
                'flex-shrink-0',
                'transition-all duration-700'
              )}
            >
              {/* Atom orbits */}
              <g stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.8">
                <ellipse cx="32" cy="32" rx="24" ry="8" />
                <ellipse cx="32" cy="32" rx="24" ry="8" transform="rotate(60 32 32)" />
                <ellipse cx="32" cy="32" rx="24" ry="8" transform="rotate(120 32 32)" />
              </g>
              {/* Open book */}
              <g fill="currentColor">
                <path d="M 26 26 L 26 38 L 32 36 L 38 38 L 38 26 Z" opacity="0.9" />
                <line x1="32" y1="26" x2="32" y2="36" stroke="currentColor" strokeWidth="1.5" />
                <line x1="28" y1="29" x2="31" y2="29" stroke="currentColor" strokeWidth="0.5" opacity="0.5" />
                <line x1="28" y1="32" x2="31" y2="32" stroke="currentColor" strokeWidth="0.5" opacity="0.5" />
                <line x1="28" y1="35" x2="31" y2="35" stroke="currentColor" strokeWidth="0.5" opacity="0.5" />
                <line x1="33" y1="29" x2="36" y2="29" stroke="currentColor" strokeWidth="0.5" opacity="0.5" />
                <line x1="33" y1="32" x2="36" y2="32" stroke="currentColor" strokeWidth="0.5" opacity="0.5" />
                <line x1="33" y1="35" x2="36" y2="35" stroke="currentColor" strokeWidth="0.5" opacity="0.5" />
              </g>
              {/* Nucleus */}
              <circle cx="32" cy="32" r="2.5" fill="currentColor" />
              {/* Electrons */}
              <circle cx="56" cy="32" r="2" fill="currentColor" opacity="0.9" />
              <circle cx="20" cy="24" r="2" fill="currentColor" opacity="0.9" />
              <circle cx="44" cy="44" r="2" fill="currentColor" opacity="0.9" />
            </svg>
            <div>
              <h1
                className={cn(
                  'text-xl font-light tracking-tight',
                  'text-accent-900 dark:text-accent-100',
                  'transition-colors duration-700'
                )}
                style={{ fontFamily: 'var(--font-display)' }}
              >
                SUJBOT2
              </h1>
              <p className={cn(
                'text-xs font-light',
                'text-accent-500 dark:text-accent-400',
                'transition-colors duration-700'
              )}>
                Legal & Technical Document Intelligence
              </p>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          {/* Theme toggle */}
          <button
            onClick={onToggleTheme}
            {...themeHover.hoverProps}
            style={themeHover.style}
            className={cn(
              'relative p-2 rounded-lg',
              'hover:bg-accent-100 dark:hover:bg-accent-800',
              'transition-all duration-700',
              'group overflow-hidden'
            )}
            title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {/* Background glow effect */}
            <div className={cn(
              'absolute inset-0 rounded-lg opacity-0',
              'group-hover:opacity-100 transition-opacity duration-700',
              'bg-gradient-to-r from-amber-100 to-blue-100',
              'dark:from-amber-900/20 dark:to-blue-900/20'
            )} />

            {/* Icons with rotation animation */}
            <div className="relative">
              <Moon
                size={20}
                className={cn(
                  'absolute inset-0 transition-all duration-700',
                  theme === 'light'
                    ? 'opacity-100 rotate-0'
                    : 'opacity-0 -rotate-90'
                )}
              />
              <Sun
                size={20}
                className={cn(
                  'transition-all duration-700',
                  theme === 'dark'
                    ? 'opacity-100 rotate-0'
                    : 'opacity-0 rotate-90'
                )}
              />
            </div>
          </button>
        </div>
      </div>
    </header>
  );
}
