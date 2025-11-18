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
            {/* Icon - Atom + Book */}
            <svg
              width="40"
              height="40"
              viewBox="0 0 512 512"
              xmlns="http://www.w3.org/2000/svg"
              className={cn(
                'text-accent-900 dark:text-accent-100',
                'flex-shrink-0',
                'transition-all duration-700'
              )}
            >
              {/* Atom + Book */}
              <g transform="translate(256 256)" stroke="currentColor" strokeWidth="10" fill="none" strokeLinecap="round">
                {/* Orbitals */}
                <ellipse rx="185" ry="110" />
                <ellipse rx="185" ry="110" transform="rotate(60)" />
                <ellipse rx="185" ry="110" transform="rotate(-60)" />

                {/* Electrons (4 directions) */}
                <circle r="12" cx="185" cy="0" fill="currentColor" stroke="none" />
                <circle r="12" cx="-185" cy="0" fill="currentColor" stroke="none" />
                <circle r="12" cx="0" cy="-110" fill="currentColor" stroke="none" />
                <circle r="12" cx="0" cy="110" fill="currentColor" stroke="none" />

                {/* Book */}
                <g fill="none" stroke="currentColor" strokeWidth="8" strokeLinejoin="round">
                  {/* Open book outline */}
                  <path d="M -75 -42
                           L -75 42
                           Q -37 24 0 36
                           Q 37 24 75 42
                           L 75 -42
                           Q 37 -52 0 -42
                           Q -37 -52 -75 -42 Z" />
                  {/* Book spine */}
                  <line x1="0" y1="-48" x2="0" y2="38" />
                  {/* Lines on left side */}
                  <path d="M -55 -22 L -20 -18" />
                  <path d="M -55  -5 L -20   0" />
                  <path d="M -55  12 L -20  17" />
                  {/* Lines on right side */}
                  <path d="M 20 -18 L 55 -22" />
                  <path d="M 20   0 L 55  -5" />
                  <path d="M 20  17 L 55  12" />
                </g>
              </g>
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
