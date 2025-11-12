/**
 * AgentProgress - Display agent execution progress
 *
 * ChatGPT-style minimalist design with animated dots
 */

import React from 'react';
import type { AgentProgress as AgentProgressType } from '../../types';

interface AgentProgressProps {
  progress: AgentProgressType;
}

export const AgentProgress: React.FC<AgentProgressProps> = ({ progress }) => {
  const { currentAgent, currentMessage, completedAgents, activeTools } = progress;

  // Debug logging
  console.log('🔍 AgentProgress render:', {
    currentAgent,
    currentMessage,
    completedAgents: completedAgents?.length || 0,
    activeTools: activeTools?.length || 0
  });

  // Don't show anything if no agent is running
  if (!currentAgent && completedAgents.length === 0) {
    console.log('❌ AgentProgress: Not rendering (no agent)');
    return null;
  }

  console.log('✅ AgentProgress: Rendering!');

  return (
    <div style={{
      padding: '8px 12px',
      fontSize: '14px',
      color: '#6b7280',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      {currentAgent && currentMessage && (
        <>
          {/* Agent message */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <span style={{
              color: '#3b82f6',
              fontSize: '16px'
            }}>
              →
            </span>
            <span>{currentMessage}</span>
            <AnimatedDots />
          </div>

          {/* Active tools */}
          {activeTools && activeTools.length > 0 && (
            <div style={{
              marginTop: '4px',
              marginLeft: '24px',
              fontSize: '13px'
            }}>
              {activeTools.map((tool, index) => (
                <div
                  key={`${tool.tool}-${index}`}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    marginTop: '2px',
                    color: tool.status === 'running' ? '#6b7280' : tool.status === 'completed' ? '#10b981' : '#ef4444'
                  }}
                >
                  <span style={{ fontSize: '12px' }}>
                    {tool.status === 'running' ? '⋯' : tool.status === 'completed' ? '✓' : '✗'}
                  </span>
                  <span>using {tool.tool}</span>
                  {tool.status === 'running' && <AnimatedDots />}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};

/**
 * Animated dots (ChatGPT-style)
 */
const AnimatedDots: React.FC = () => {
  return (
    <span style={{
      display: 'inline-flex',
      gap: '2px'
    }}>
      <Dot delay={0} />
      <Dot delay={0.2} />
      <Dot delay={0.4} />
    </span>
  );
};

const Dot: React.FC<{ delay: number }> = ({ delay }) => {
  const [opacity, setOpacity] = React.useState(0.3);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setOpacity(prev => {
        const nextValue = prev + 0.1;
        return nextValue > 1 ? 0.3 : nextValue;
      });
    }, 140);

    // Stagger animation start
    const timeout = setTimeout(() => {
      clearInterval(interval);
      const newInterval = setInterval(() => {
        setOpacity(prev => {
          const nextValue = prev + 0.1;
          return nextValue > 1 ? 0.3 : nextValue;
        });
      }, 140);
      return () => clearInterval(newInterval);
    }, delay * 1000);

    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
  }, [delay]);

  return (
    <span style={{
      opacity,
      transition: 'opacity 0.14s ease-in-out'
    }}>
      .
    </span>
  );
};
