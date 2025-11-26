import { useState, useEffect, useRef, useLayoutEffect } from 'react';
import { Zap, Server } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';

type Variant = 'premium' | 'local';

interface VariantInfo {
  variant: Variant;
  display_name: string;
  model: string;
}

interface IndicatorStyle {
  left: number;
  width: number;
}

export function AgentVariantSelector() {
  const [currentVariant, setCurrentVariant] = useState<Variant>('premium');
  const [isLoading, setIsLoading] = useState(true);
  const [isSwitching, setIsSwitching] = useState(false);
  const [indicatorStyle, setIndicatorStyle] = useState<IndicatorStyle>({ left: 4, width: 80 });

  const premiumRef = useRef<HTMLButtonElement>(null);
  const localRef = useRef<HTMLButtonElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadVariant();
  }, []);

  // Measure button positions and update indicator
  useLayoutEffect(() => {
    const updateIndicator = () => {
      const activeRef = currentVariant === 'premium' ? premiumRef : localRef;
      const button = activeRef.current;
      const container = containerRef.current;

      if (button && container) {
        const containerRect = container.getBoundingClientRect();
        const buttonRect = button.getBoundingClientRect();
        setIndicatorStyle({
          left: buttonRect.left - containerRect.left,
          width: buttonRect.width,
        });
      }
    };

    updateIndicator();
    window.addEventListener('resize', updateIndicator);
    return () => window.removeEventListener('resize', updateIndicator);
  }, [currentVariant, isLoading]);

  const loadVariant = async () => {
    try {
      const response = await fetch('/settings/agent-variant', {
        credentials: 'include'
      });

      if (response.ok) {
        const data: VariantInfo = await response.json();
        setCurrentVariant(data.variant);
      }
    } catch (error) {
      console.error('Failed to load variant:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const switchVariant = async (variant: Variant) => {
    if (variant === currentVariant || isSwitching) return;

    setIsSwitching(true);
    try {
      const response = await fetch('/settings/agent-variant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ variant })
      });

      if (response.ok) {
        setCurrentVariant(variant);
      } else {
        console.error('Failed to switch variant');
      }
    } catch (error) {
      console.error('Error switching variant:', error);
    } finally {
      setIsSwitching(false);
    }
  };

  if (isLoading) {
    return (
      <div className="h-9 w-40 bg-accent-200 dark:bg-accent-700 rounded-lg animate-pulse" />
    );
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        'relative flex items-center p-1 rounded-lg',
        'bg-accent-100',
        'border border-accent-200'
      )}
    >
      {/* Sliding indicator */}
      <div
        className={cn(
          'absolute top-1 bottom-1 rounded-md',
          'bg-white shadow-sm',
          'transition-all duration-300 ease-out'
        )}
        style={{
          left: `${indicatorStyle.left}px`,
          width: `${indicatorStyle.width}px`,
        }}
      />

      {/* Premium Button */}
      <button
        ref={premiumRef}
        onClick={() => switchVariant('premium')}
        disabled={isSwitching}
        className={cn(
          'relative z-10 flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium',
          'transition-colors duration-300',
          currentVariant === 'premium'
            ? 'text-accent-900'
            : 'text-accent-500 hover:text-accent-700',
          isSwitching && 'opacity-50 cursor-not-allowed'
        )}
        title="Premium - Claude Haiku 4.5 (rychlé, kvalitní)"
      >
        <Zap
          size={16}
          className={cn(
            'transition-colors duration-300',
            currentVariant === 'premium' ? 'text-yellow-500' : ''
          )}
        />
        <span>Premium</span>
      </button>

      {/* Local Button */}
      <button
        ref={localRef}
        onClick={() => switchVariant('local')}
        disabled={isSwitching}
        className={cn(
          'relative z-10 flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium',
          'transition-colors duration-300',
          currentVariant === 'local'
            ? 'text-accent-900'
            : 'text-accent-500 hover:text-accent-700',
          isSwitching && 'opacity-50 cursor-not-allowed'
        )}
        title="Local - Llama 3.1 70B (open-source přes DeepInfra)"
      >
        <Server
          size={16}
          className={cn(
            'transition-colors duration-300',
            currentVariant === 'local' ? 'text-blue-500' : ''
          )}
        />
        <span>Local</span>
      </button>
    </div>
  );
}
