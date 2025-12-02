import { useState, useEffect, useRef, useLayoutEffect } from 'react';
import { DollarSign, Server } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { useAuth } from '../../contexts/AuthContext';

type Variant = 'premium' | 'cheap' | 'local';

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
  const { t } = useTranslation();
  const { user } = useAuth();
  const isAdmin = user?.is_admin ?? false;

  const [currentVariant, setCurrentVariant] = useState<Variant>('cheap');
  const [isLoading, setIsLoading] = useState(true);
  const [isSwitching, setIsSwitching] = useState(false);
  const [indicatorStyle, setIndicatorStyle] = useState<IndicatorStyle>({ left: 4, width: 80 });

  const premiumRef = useRef<HTMLButtonElement>(null);
  const cheapRef = useRef<HTMLButtonElement>(null);
  const localRef = useRef<HTMLButtonElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadVariant();
  }, []);

  // Measure button positions and update indicator
  useLayoutEffect(() => {
    const updateIndicator = () => {
      const refs: Record<Variant, React.RefObject<HTMLButtonElement | null>> = {
        premium: premiumRef,
        cheap: cheapRef,
        local: localRef,
      };
      const activeRef = refs[currentVariant];
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
      <div className="h-9 w-56 bg-accent-200 dark:bg-accent-700 rounded-lg animate-pulse" />
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

      {/* Premium Button - Only visible to admins */}
      {isAdmin && (
        <button
          ref={premiumRef}
          onClick={() => switchVariant('premium')}
          disabled={isSwitching}
          className={cn(
            'relative z-10 flex items-center gap-1 px-3 py-1.5 rounded-md text-sm font-medium',
            'transition-colors duration-300',
            currentVariant === 'premium'
              ? 'text-accent-900'
              : 'text-accent-500 hover:text-accent-700',
            isSwitching && 'opacity-50 cursor-not-allowed'
          )}
          title={t('agentVariant.premiumTooltip')}
        >
          <span
            className={cn(
              'flex -space-x-1.5 transition-colors duration-300',
              currentVariant === 'premium' ? 'text-yellow-500' : ''
            )}
          >
            <DollarSign size={14} />
            <DollarSign size={14} />
            <DollarSign size={14} />
          </span>
          <span>{t('agentVariant.premium')}</span>
        </button>
      )}

      {/* Cheap Button */}
      <button
        ref={cheapRef}
        onClick={() => switchVariant('cheap')}
        disabled={isSwitching}
        className={cn(
          'relative z-10 flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium',
          'transition-colors duration-300',
          currentVariant === 'cheap'
            ? 'text-accent-900'
            : 'text-accent-500 hover:text-accent-700',
          isSwitching && 'opacity-50 cursor-not-allowed'
        )}
        title={t('agentVariant.cheapTooltip')}
      >
        <DollarSign
          size={16}
          className={cn(
            'transition-colors duration-300',
            currentVariant === 'cheap' ? 'text-green-500' : ''
          )}
        />
        <span>{t('agentVariant.cheap')}</span>
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
        title={t('agentVariant.localTooltip')}
      >
        <Server
          size={16}
          className={cn(
            'transition-colors duration-300',
            currentVariant === 'local' ? 'text-blue-500' : ''
          )}
        />
        <span>{t('agentVariant.local')}</span>
      </button>
    </div>
  );
}
