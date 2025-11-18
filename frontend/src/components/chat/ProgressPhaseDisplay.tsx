/**
 * ProgressPhaseDisplay - Elegant phase-based progress visualization
 *
 * Design Philosophy: Refined minimalist with sophisticated typography
 * - Cormorant Garamond for phase names (elegant serif)
 * - Inter for tool names (clean sans-serif)
 * - Monochromatic palette with subtle accent colors
 * - Smooth transitions and micro-interactions
 */

import React from 'react';
import { cn } from '../../design-system/utils/cn';
import type { AgentProgress as AgentProgressType } from '../../types';

interface ProgressPhaseDisplayProps {
  progress: AgentProgressType;
}

// Phase definitions with visual metadata
interface PhaseConfig {
  name: string;
  icon: string;
  description: string;
  color: string; // Tailwind color class
  darkColor: string;
}

const PHASE_MAP: Record<string, PhaseConfig> = {
  orchestrator: {
    name: 'Planning',
    icon: '⚡',
    description: 'Analyzing query and planning approach',
    color: 'text-purple-600',
    darkColor: 'dark:text-purple-400',
  },
  extractor: {
    name: 'Searching',
    icon: '🔍',
    description: 'Retrieving relevant documents',
    color: 'text-blue-600',
    darkColor: 'dark:text-blue-400',
  },
  classifier: {
    name: 'Classifying',
    icon: '🏷️',
    description: 'Categorizing content',
    color: 'text-green-600',
    darkColor: 'dark:text-green-400',
  },
  compliance: {
    name: 'Compliance Check',
    icon: '✓',
    description: 'Verifying regulatory compliance',
    color: 'text-amber-600',
    darkColor: 'dark:text-amber-400',
  },
  risk_verifier: {
    name: 'Risk Assessment',
    icon: '⚠️',
    description: 'Analyzing potential risks',
    color: 'text-red-600',
    darkColor: 'dark:text-red-400',
  },
  citation_auditor: {
    name: 'Citation Audit',
    icon: '📝',
    description: 'Verifying citations and sources',
    color: 'text-indigo-600',
    darkColor: 'dark:text-indigo-400',
  },
  gap_synthesizer: {
    name: 'Synthesizing',
    icon: '🔗',
    description: 'Combining information from sources',
    color: 'text-teal-600',
    darkColor: 'dark:text-teal-400',
  },
  report_generator: {
    name: 'Generating Report',
    icon: '📄',
    description: 'Creating final response',
    color: 'text-violet-600',
    darkColor: 'dark:text-violet-400',
  },
};

// Fallback for unknown agents
const DEFAULT_PHASE: PhaseConfig = {
  name: 'Processing',
  icon: '⚙️',
  description: 'Working on your request',
  color: 'text-gray-600',
  darkColor: 'dark:text-gray-400',
};

export const ProgressPhaseDisplay: React.FC<ProgressPhaseDisplayProps> = ({ progress }) => {
  const { currentAgent, currentMessage, activeTools } = progress;

  // Only show during active generation (when there's a current agent)
  // Hide completely when generation is done
  if (!currentAgent) {
    return null;
  }

  // Get phase config
  const phaseConfig = currentAgent ? (PHASE_MAP[currentAgent] || DEFAULT_PHASE) : DEFAULT_PHASE;

  return (
    <div
      className={cn(
        'relative overflow-hidden',
        'rounded-xl',
        'bg-gradient-to-br from-white to-gray-50',
        'dark:from-gray-900 dark:to-gray-800',
        'border border-gray-200 dark:border-gray-700',
        'shadow-sm',
        'transition-all duration-500 ease-out'
      )}
      style={{
        animation: 'fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
      }}
    >
      {/* Animated background shimmer */}
      <div
        className={cn(
          'absolute inset-0 opacity-30',
          'bg-gradient-to-r from-transparent via-gray-100 dark:via-gray-700 to-transparent',
        )}
        style={{
          animation: 'shimmer 2s infinite linear',
          backgroundSize: '200% 100%',
        }}
      />

      <div className="relative px-5 py-4">
        {/* Phase header */}
        {currentAgent && (
          <div className="flex items-start gap-4">
            {/* Icon with pulse animation */}
            <div
              className={cn(
                'flex-shrink-0',
                'text-2xl',
                'animate-pulse'
              )}
              style={{
                animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
              }}
            >
              {phaseConfig.icon}
            </div>

            {/* Phase info */}
            <div className="flex-1 min-w-0">
              {/* Phase name (Cormorant Garamond - elegant serif) */}
              <h3
                className={cn(
                  'text-lg font-semibold tracking-wide',
                  'font-display', // Cormorant Garamond
                  phaseConfig.color,
                  phaseConfig.darkColor,
                  'mb-1'
                )}
              >
                {phaseConfig.name}
              </h3>

              {/* Phase description */}
              <p
                className={cn(
                  'text-sm',
                  'text-gray-600 dark:text-gray-400',
                  'mb-3'
                )}
              >
                {currentMessage || phaseConfig.description}
              </p>

              {/* Loading bar */}
              <div
                className={cn(
                  'h-1 bg-gray-200 dark:bg-gray-700',
                  'rounded-full overflow-hidden'
                )}
              >
                <div
                  className={cn(
                    'h-full',
                    'bg-gradient-to-r from-blue-500 to-purple-500',
                    'dark:from-blue-400 dark:to-purple-400',
                  )}
                  style={{
                    animation: 'indeterminateProgress 1.5s ease-in-out infinite',
                    width: '40%',
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Active tools section */}
        {activeTools && activeTools.length > 0 && (
          <div
            className={cn(
              'mt-4 pt-4',
              'border-t border-gray-200 dark:border-gray-700',
              'space-y-2'
            )}
          >
            <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
              Active Tools
            </div>

            {activeTools.map((tool, index) => (
              <div
                key={`${tool.tool}-${index}`}
                className={cn(
                  'flex items-center gap-3',
                  'text-sm',
                  'transition-all duration-300'
                )}
                style={{
                  animation: `fadeInUp 0.3s cubic-bezier(0.16, 1, 0.3, 1) ${index * 50}ms backwards`,
                }}
              >
                {/* Status indicator */}
                <div
                  className={cn(
                    'flex-shrink-0 w-2 h-2 rounded-full',
                    tool.status === 'running'
                      ? 'bg-blue-500 dark:bg-blue-400 animate-pulse'
                      : tool.status === 'completed'
                      ? 'bg-green-500 dark:bg-green-400'
                      : 'bg-red-500 dark:bg-red-400'
                  )}
                />

                {/* Tool name */}
                <span
                  className={cn(
                    'flex-1',
                    tool.status === 'running'
                      ? 'text-gray-900 dark:text-gray-100'
                      : tool.status === 'completed'
                      ? 'text-gray-600 dark:text-gray-400'
                      : 'text-red-600 dark:text-red-400'
                  )}
                >
                  {tool.tool}
                </span>

                {/* Status icon */}
                <span className="flex-shrink-0 text-xs">
                  {tool.status === 'running' ? (
                    <SpinningLoader />
                  ) : tool.status === 'completed' ? (
                    '✓'
                  ) : (
                    '✗'
                  )}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Spinning loader component (minimal, elegant)
 */
const SpinningLoader: React.FC = () => {
  return (
    <div
      className={cn(
        'inline-block w-3 h-3',
        'border-2 border-gray-300 dark:border-gray-600',
        'border-t-blue-500 dark:border-t-blue-400',
        'rounded-full'
      )}
      style={{
        animation: 'spin 0.8s linear infinite',
      }}
    />
  );
};

// Add keyframe animations to global CSS (or use Tailwind's arbitrary variants)
// For now, using inline styles with animation names that should be defined in index.css
