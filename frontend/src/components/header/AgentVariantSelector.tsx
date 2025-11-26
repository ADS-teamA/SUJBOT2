import { useState, useEffect } from 'react';
import { Zap, Server, Check } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';

type Variant = 'premium' | 'local';

interface VariantInfo {
  variant: Variant;
  display_name: string;
  model: string;
}

export function AgentVariantSelector() {
  const [currentVariant, setCurrentVariant] = useState<Variant>('premium');
  const [isLoading, setIsLoading] = useState(true);
  const [isSwitching, setIsSwitching] = useState(false);

  useEffect(() => {
    loadVariant();
  }, []);

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
      className={cn(
        'flex items-center gap-1 p-1 rounded-lg',
        'bg-accent-100 dark:bg-accent-800',
        'border border-accent-200 dark:border-accent-700'
      )}
    >
      {/* Premium Button */}
      <button
        onClick={() => switchVariant('premium')}
        disabled={isSwitching}
        className={cn(
          'flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all',
          currentVariant === 'premium'
            ? 'bg-white dark:bg-accent-900 text-accent-900 dark:text-accent-100 shadow-sm'
            : 'text-accent-600 dark:text-accent-400 hover:text-accent-900 dark:hover:text-accent-100',
          isSwitching && 'opacity-50 cursor-not-allowed'
        )}
        title="Premium - Claude Haiku 4.5 (rychlé, kvalitní)"
      >
        <Zap
          size={16}
          className={currentVariant === 'premium' ? 'text-yellow-500' : ''}
        />
        <span>Premium</span>
        {currentVariant === 'premium' && <Check size={14} className="text-green-600" />}
      </button>

      {/* Local Button */}
      <button
        onClick={() => switchVariant('local')}
        disabled={isSwitching}
        className={cn(
          'flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all',
          currentVariant === 'local'
            ? 'bg-white dark:bg-accent-900 text-accent-900 dark:text-accent-100 shadow-sm'
            : 'text-accent-600 dark:text-accent-400 hover:text-accent-900 dark:hover:text-accent-100',
          isSwitching && 'opacity-50 cursor-not-allowed'
        )}
        title="Local - Llama 3.1 70B (open-source přes DeepInfra)"
      >
        <Server
          size={16}
          className={currentVariant === 'local' ? 'text-blue-500' : ''}
        />
        <span>Local</span>
        {currentVariant === 'local' && <Check size={14} className="text-green-600" />}
      </button>
    </div>
  );
}
