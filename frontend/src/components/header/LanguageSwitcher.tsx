/**
 * LanguageSwitcher Component - CZ/EN toggle switch
 *
 * Design inspired by AgentVariantSelector:
 * - Sliding indicator for active language
 * - Smooth transitions matching app theme (300ms)
 * - Compact toggle format
 */

import { useState, useRef, useLayoutEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';

type Language = 'cs' | 'en';

interface IndicatorStyle {
  left: number;
  width: number;
}

export function LanguageSwitcher() {
  const { i18n, t } = useTranslation();
  const [indicatorStyle, setIndicatorStyle] = useState<IndicatorStyle>({ left: 4, width: 40 });

  const csRef = useRef<HTMLButtonElement>(null);
  const enRef = useRef<HTMLButtonElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Normalize language code (handle 'cs-CZ' -> 'cs', 'en-US' -> 'en')
  const currentLanguage: Language = i18n.language?.startsWith('cs') ? 'cs' : 'en';

  // Measure button positions and update indicator
  useLayoutEffect(() => {
    const updateIndicator = () => {
      const activeRef = currentLanguage === 'cs' ? csRef : enRef;
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
  }, [currentLanguage]);

  const switchLanguage = (language: Language) => {
    if (language !== currentLanguage) {
      i18n.changeLanguage(language);
    }
  };

  return (
    <div
      ref={containerRef}
      className={cn(
        'relative flex items-center p-1 rounded-lg',
        'bg-accent-100',
        'border border-accent-200'
      )}
      title={t('language.switchTo')}
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

      {/* CZ Button */}
      <button
        ref={csRef}
        onClick={() => switchLanguage('cs')}
        className={cn(
          'relative z-10 px-3 py-1.5 rounded-md text-sm font-medium',
          'transition-colors duration-300',
          currentLanguage === 'cs'
            ? 'text-accent-900'
            : 'text-accent-500 hover:text-accent-700'
        )}
        aria-label={t('language.czech')}
      >
        CZ
      </button>

      {/* EN Button */}
      <button
        ref={enRef}
        onClick={() => switchLanguage('en')}
        className={cn(
          'relative z-10 px-3 py-1.5 rounded-md text-sm font-medium',
          'transition-colors duration-300',
          currentLanguage === 'en'
            ? 'text-accent-900'
            : 'text-accent-500 hover:text-accent-700'
        )}
        aria-label={t('language.english')}
      >
        EN
      </button>
    </div>
  );
}
