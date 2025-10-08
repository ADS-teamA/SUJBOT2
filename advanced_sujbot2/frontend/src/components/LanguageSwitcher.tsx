import React from 'react';
import { Switch } from '@/components/ui/switch';
import { useUIStore } from '@/stores/uiStore';
import { cn } from '@/utils/cn';

export const LanguageSwitcher: React.FC = () => {
  const { language, toggleLanguage } = useUIStore();

  return (
    <div className="flex items-center gap-2">
      <span className={cn('text-sm', language === 'cs' && 'font-bold')}>
        CZ
      </span>
      <Switch
        checked={language === 'en'}
        onCheckedChange={toggleLanguage}
      />
      <span className={cn('text-sm', language === 'en' && 'font-bold')}>
        EN
      </span>
    </div>
  );
};
