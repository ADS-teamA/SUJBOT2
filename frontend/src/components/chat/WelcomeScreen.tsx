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
}

const SUGGESTED_PROMPTS = [
  {
    icon: Scale,
    title: 'Legal Compliance',
    prompt: 'What are the GDPR compliance requirements for data processing?',
  },
  {
    icon: Shield,
    title: 'Risk Assessment',
    prompt: 'Analyze the cybersecurity risks in our data retention policy',
  },
  {
    icon: FileCheck,
    title: 'Document Comparison',
    prompt: 'Compare the privacy policies across our documents',
  },
  {
    icon: FileText,
    title: 'Citation Lookup',
    prompt: 'Find all references to data protection regulations',
  },
];

export function WelcomeScreen({ onPromptClick }: WelcomeScreenProps) {
  return (
    <div className={cn(
      'flex-1 flex flex-col items-center justify-start',
      'px-6 pt-8 pb-32 overflow-hidden'
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
        className="max-w-4xl w-full flex flex-col"
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
              viewBox="0 0 64 64"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className={cn(
                'text-accent-900 dark:text-accent-100',
                'opacity-90'
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

        {/* Spacer for centered input box (ChatInput is absolutely positioned here) */}
        <div className="-mt-4" />

        {/* Suggested prompts */}
        <div className="space-y-3 mt-44">
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

        {/* Capabilities hint */}
        <div className="text-center space-y-1 mt-6">
          <p className={cn(
            'text-xs',
            'text-accent-400 dark:text-accent-600'
          )}>
            Multi-agent RAG system with 7 specialized agents
          </p>
          <p className={cn(
            'text-xs',
            'text-accent-400 dark:text-accent-600'
          )}>
            Hierarchical document analysis • Knowledge graph integration • Citation verification
          </p>
        </div>
      </div>
    </div>
  );
}
