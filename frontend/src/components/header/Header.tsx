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
      'px-6 py-4'
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
              'transition-colors duration-150'
            )}
            aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          >
            <Menu size={20} />
          </button>

          {/* Logo and title */}
          <div className="flex items-center gap-3">
            <div className={cn(
              'w-8 h-8 rounded-lg',
              'bg-accent-700 dark:bg-accent-300',
              'flex items-center justify-center',
              'text-white dark:text-accent-900',
              'font-bold'
            )}>
              S2
            </div>
            <div>
              <h1 className={cn(
                'text-lg font-bold',
                'text-accent-900 dark:text-accent-100'
              )}>SUJBOT2</h1>
              <p className={cn(
                'text-xs',
                'text-accent-500 dark:text-accent-400'
              )}>
                RAG-Powered Document Assistant
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
              'p-2 rounded-lg',
              'hover:bg-accent-100 dark:hover:bg-accent-800',
              'transition-colors duration-150'
            )}
            title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
          </button>
        </div>
      </div>
    </header>
  );
}
