/**
 * Header Component - Top navigation with sidebar control
 */

import { Menu, LogOut } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';
import { useHover } from '../../design-system/animations/hooks/useHover';
import { useAuth } from '../../contexts/AuthContext';
import { AgentVariantSelector } from './AgentVariantSelector';

interface HeaderProps {
  onToggleSidebar: () => void;
  sidebarOpen: boolean;
}

export function Header({
  onToggleSidebar,
  sidebarOpen,
}: HeaderProps) {
  // Authentication
  const { logout } = useAuth();

  // Animation hooks
  const hamburgerHover = useHover({ scale: true });
  const logoutHover = useHover({ scale: true });

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
              <g transform="translate(256 256)" stroke="currentColor" fill="none" strokeLinecap="round">
                {/* Orbitals (thicker) */}
                <ellipse rx="185" ry="110" strokeWidth="16" />
                <ellipse rx="185" ry="110" strokeWidth="16" transform="rotate(60)" />
                <ellipse rx="185" ry="110" strokeWidth="16" transform="rotate(-60)" />

                {/* Electrons (3 atoms evenly distributed at 0°, 120°, 240°) */}
                <circle r="20" cx="185" cy="0" fill="currentColor" stroke="none" />
                <circle r="20" cx="-92.5" cy="160" fill="currentColor" stroke="none" />
                <circle r="20" cx="-92.5" cy="-160" fill="currentColor" stroke="none" />

                {/* Paragraph symbol § */}
                <text
                  x="0"
                  y="35"
                  fontSize="140"
                  fontWeight="bold"
                  fill="currentColor"
                  textAnchor="middle"
                  fontFamily="serif"
                >§</text>
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
          {/* Agent variant selector */}
          <AgentVariantSelector />

          {/* Logout button */}
          <button
            onClick={logout}
            {...logoutHover.hoverProps}
            style={logoutHover.style}
            className={cn(
              'p-2 rounded-lg',
              'text-accent-700 dark:text-accent-300',
              'hover:bg-accent-100 dark:hover:bg-accent-800',
              'transition-all duration-700'
            )}
            title="Sign out"
          >
            <LogOut size={20} className="transition-all duration-700" />
          </button>
        </div>
      </div>
    </header>
  );
}
