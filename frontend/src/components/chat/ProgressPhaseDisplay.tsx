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
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import type { AgentProgress as AgentProgressType } from '../../types';

interface ProgressPhaseDisplayProps {
  progress: AgentProgressType;
}

// Phase definitions with visual metadata and translation keys
interface PhaseConfig {
  nameKey: string;
  descKey: string;
  icon: string;
  color: string; // Tailwind color class
  darkColor: string;
}

const PHASE_MAP: Record<string, PhaseConfig> = {
  orchestrator: {
    nameKey: 'progress.planning',
    descKey: 'progress.planningDesc',
    icon: '⚡',
    color: 'text-purple-600',
    darkColor: 'dark:text-purple-400',
  },
  extractor: {
    nameKey: 'progress.searching',
    descKey: 'progress.searchingDesc',
    icon: '🔍',
    color: 'text-blue-600',
    darkColor: 'dark:text-blue-400',
  },
  classifier: {
    nameKey: 'progress.classifying',
    descKey: 'progress.classifyingDesc',
    icon: '🏷️',
    color: 'text-green-600',
    darkColor: 'dark:text-green-400',
  },
  compliance: {
    nameKey: 'progress.complianceCheck',
    descKey: 'progress.complianceCheckDesc',
    icon: '✓',
    color: 'text-amber-600',
    darkColor: 'dark:text-amber-400',
  },
  risk_verifier: {
    nameKey: 'progress.riskAssessment',
    descKey: 'progress.riskAssessmentDesc',
    icon: '⚠️',
    color: 'text-red-600',
    darkColor: 'dark:text-red-400',
  },
  citation_auditor: {
    nameKey: 'progress.citationAudit',
    descKey: 'progress.citationAuditDesc',
    icon: '📝',
    color: 'text-indigo-600',
    darkColor: 'dark:text-indigo-400',
  },
  gap_synthesizer: {
    nameKey: 'progress.synthesizing',
    descKey: 'progress.synthesizingDesc',
    icon: '🔗',
    color: 'text-teal-600',
    darkColor: 'dark:text-teal-400',
  },
  report_generator: {
    nameKey: 'progress.generatingReport',
    descKey: 'progress.generatingReportDesc',
    icon: '📄',
    color: 'text-violet-600',
    darkColor: 'dark:text-violet-400',
  },
};

// Fallback for unknown agents
const DEFAULT_PHASE: PhaseConfig = {
  nameKey: 'progress.processing',
  descKey: 'progress.processingDesc',
  icon: '⚙️',
  color: 'text-gray-600',
  darkColor: 'dark:text-gray-400',
};

export const ProgressPhaseDisplay: React.FC<ProgressPhaseDisplayProps> = ({ progress }) => {
  const { t } = useTranslation();
  const { currentAgent, currentMessage, activeTools } = progress;

  // Only show during active generation (when there's a current agent)
  // Hide completely when generation is done
  if (!currentAgent) {
    return null;
  }

  // Get phase config
  const phaseConfig = currentAgent ? (PHASE_MAP[currentAgent] || DEFAULT_PHASE) : DEFAULT_PHASE;
  const phaseName = t(phaseConfig.nameKey);
  const phaseDescription = t(phaseConfig.descKey);

  // Translate backend messages that come in English
  const getDisplayMessage = (message: string | null | undefined): string => {
    if (!message) return phaseDescription;

    // Map backend messages to translation keys
    const messageMap: Record<string, string> = {
      'searching documents': 'progress.searchingDocuments',
      'classifying query': 'progress.classifyingQuery',
      'checking compliance': 'progress.checkingCompliance',
      'verifying risks': 'progress.verifyingRisks',
      'auditing citations': 'progress.auditingCitations',
      'synthesizing information': 'progress.synthesizingInfo',
      'generating report': 'progress.generatingReportMsg',
    };

    const lowerMessage = message.toLowerCase();

    // Check for exact matches first
    if (messageMap[lowerMessage]) {
      return t(messageMap[lowerMessage]);
    }

    // Check for "Initializing..." messages
    if (lowerMessage.includes('initializing')) {
      return t('progress.initializing');
    }

    // Check for "Running X" pattern
    const runningMatch = message.match(/^Running\s+(.+)$/i);
    if (runningMatch) {
      return t('progress.runningAgent', { agent: runningMatch[1] });
    }

    return message;
  };

  return (
    <div
      className={cn(
        'relative',
        'transition-all duration-500 ease-out'
      )}
      style={{
        animation: 'fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
      }}
    >

      <div className="relative">
        {/* Phase header and Icon - Compact Row */}
        {currentAgent && (
          <div className="mb-2">
            <div className="flex items-center gap-3 mb-2">
              {/* Icon */}
              <div
                className={cn(
                  'flex-shrink-0',
                  'text-xl', // Smaller icon
                  'animate-pulse'
                )}
                style={{
                  animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                }}
              >
                {phaseConfig.icon}
              </div>

              {/* Text Info */}
              <div className="flex items-baseline gap-2 min-w-0 overflow-hidden">
                <h3
                  className={cn(
                    'text-base font-semibold tracking-wide truncate', // Smaller title
                    'font-display',
                    phaseConfig.color,
                    phaseConfig.darkColor
                  )}
                >
                  {phaseName}
                </h3>

                <p
                  className={cn(
                    'text-xs truncate',
                    'text-gray-500 dark:text-gray-500'
                  )}
                >
                  {getDisplayMessage(currentMessage)}
                </p>
              </div>
            </div>

            {/* Loading bar - Full Width & Ultra Thin */}
            <div
              className={cn(
                'h-[1px] w-full bg-gray-100 dark:bg-gray-800', // Ultra thin track
                'overflow-hidden'
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
        )}

        {/* Active tools section */}
        {activeTools && activeTools.length > 0 && (
          <div
            className={cn(
              'mt-2 pt-2',
              'border-t border-gray-100 dark:border-gray-800',
              'space-y-1'
            )}
          >
            <div className="text-[10px] font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wider mb-1">
              {t('progress.activeTools')}
            </div>

            {activeTools.map((tool, index) => (
              <div
                key={`${tool.tool}-${index}`}
                className={cn(
                  'flex items-center gap-2',
                  'text-xs',
                  'transition-all duration-300'
                )}
                style={{
                  animation: `fadeInUp 0.3s cubic-bezier(0.16, 1, 0.3, 1) ${index * 50}ms backwards`,
                }}
              >
                {/* Status indicator */}
                <div
                  className={cn(
                    'flex-shrink-0 w-1.5 h-1.5 rounded-full',
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
                    'flex-1 truncate',
                    tool.status === 'running'
                      ? 'text-gray-900 dark:text-gray-100'
                      : tool.status === 'completed'
                        ? 'text-gray-500 dark:text-gray-500'
                        : 'text-red-600 dark:text-red-400'
                  )}
                >
                  {tool.tool}
                </span>

                {/* Status icon */}
                <span className="flex-shrink-0 text-[10px]">
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
