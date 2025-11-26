/**
 * WelcomeScreen Component - Displayed when starting a new conversation
 *
 * Features:
 * - Large SUJBOT2 branding with serif typography
 * - Suggested prompts users can click
 * - Gradient background effects
 * - Smooth fade-in animations
 */

import { FileText, Scale, Shield, FileCheck } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';

interface WelcomeScreenProps {
  onPromptClick: (prompt: string) => void;
  children?: React.ReactNode;
}

const SUGGESTED_PROMPTS = [
  {
    icon: Scale,
    title: 'Regulatory Compliance',
    prompt: 'Jaké jsou požadavky SÚJB pro provoz jaderných zařízení?',
  },
  {
    icon: Shield,
    title: 'Safety Analysis',
    prompt: 'Analyzuj bezpečnostní opatření pro nakládání s radioaktivním odpadem',
  },
  {
    icon: FileCheck,
    title: 'Document Comparison',
    prompt: 'Porovnej požadavky různých vyhlášek SÚJB',
  },
  {
    icon: FileText,
    title: 'Citation Lookup',
    prompt: 'Najdi všechny reference na atomový zákon č. 263/2016 Sb.',
  },
];

export function WelcomeScreen({ onPromptClick, children }: WelcomeScreenProps) {
  return (
    <div className={cn(
      'flex-1 flex flex-col items-center justify-center',
      'px-6 py-8 overflow-hidden'
    )}>
      {/* Gradient background */}
      <div
        className={cn(
          'absolute inset-0 -z-10',
          'bg-white dark:bg-accent-950'
        )}
        style={{
          background: 'var(--gradient-mesh-light)',
        }}
      />
      <div
        className={cn(
          'absolute inset-0 -z-10',
          'dark:block hidden'
        )}
        style={{
          background: 'var(--gradient-mesh-dark)',
        }}
      />
      <div
        className={cn(
          'absolute inset-0 -z-10'
        )}
        style={{
          background: 'var(--gradient-light)',
        }}
      />
      <div
        className={cn(
          'absolute inset-0 -z-10',
          'dark:block hidden'
        )}
        style={{
          background: 'var(--gradient-dark)',
        }}
      />

      {/* Main content */}
      <div
        className="max-w-4xl w-full flex flex-col items-center gap-8"
        style={{
          animation: 'fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        {/* Branding */}
        <div className="text-center space-y-2">
          {/* Icon */}
          <div className="flex justify-center mb-1">
            <svg
              width="64"
              height="64"
              viewBox="0 0 512 512"
              xmlns="http://www.w3.org/2000/svg"
              className={cn(
                'text-accent-900 dark:text-accent-100',
                'opacity-90'
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
          </div>

          <h1
            className={cn(
              'text-6xl font-light tracking-tight',
              'text-accent-950 dark:text-accent-50'
            )}
            style={{ fontFamily: 'var(--font-display)' }}
          >
            SUJBOT2
          </h1>
        </div>

        {/* Search bar slot */}
        {children && (
          <div className="w-full max-w-3xl">
            {children}
          </div>
        )}

        {/* Suggested prompts */}
        <div className="w-full space-y-3">
          <p className={cn(
            'text-sm font-medium tracking-wide uppercase',
            'text-accent-500 dark:text-accent-500',
            'text-center'
          )}>
            Suggested Questions
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {SUGGESTED_PROMPTS.map((item, index) => {
              const Icon = item.icon;
              return (
                <button
                  key={index}
                  onClick={() => onPromptClick(item.prompt)}
                  className={cn(
                    'group relative',
                    'p-4 rounded-xl',
                    'border border-accent-200 dark:border-accent-800',
                    'bg-white/80 dark:bg-accent-900/50',
                    'backdrop-blur-sm',
                    'hover:border-accent-400 dark:hover:border-accent-600',
                    'hover:shadow-lg',
                    'transition-all duration-300',
                    'text-left'
                  )}
                  style={{
                    animation: `fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) ${index * 0.1 + 0.2}s backwards`,
                  }}
                >
                  <div className="flex items-start gap-3">
                    <div className={cn(
                      'flex-shrink-0 w-10 h-10 rounded-lg',
                      'bg-accent-100 dark:bg-accent-800',
                      'flex items-center justify-center',
                      'group-hover:bg-accent-200 dark:group-hover:bg-accent-700',
                      'transition-colors duration-300'
                    )}>
                      <Icon size={20} className="text-accent-700 dark:text-accent-300" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className={cn(
                        'text-sm font-medium mb-1',
                        'text-accent-900 dark:text-accent-100'
                      )}>
                        {item.title}
                      </div>
                      <div className={cn(
                        'text-xs line-clamp-2',
                        'text-accent-600 dark:text-accent-400'
                      )}>
                        {item.prompt}
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

      </div>
    </div>
  );
}
