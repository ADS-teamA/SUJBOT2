import React from 'react';
import { Scale } from 'lucide-react';
import { LanguageSwitcher } from './LanguageSwitcher';
import { ThemeToggle } from './ThemeToggle';

export const TopBar: React.FC = () => {
  return (
    <div className="h-16 bg-gradient-to-r from-white via-gray-50 to-white dark:from-black dark:via-gray-950 dark:to-black border-b border-gray-200 dark:border-gray-800 px-6 flex items-center justify-between backdrop-blur-sm transition-all duration-300">
      <div className="flex items-center gap-3 animate-fade-in">
        <div className="relative group">
          <div className="absolute inset-0 bg-gradient-to-r from-gray-400 to-gray-600 dark:from-gray-300 dark:to-gray-500 rounded-lg blur opacity-25 group-hover:opacity-40 transition-opacity duration-300"></div>
          <Scale className="relative h-8 w-8 text-primary transition-transform duration-300 group-hover:scale-110" />
        </div>
        <div>
          <h1 className="text-xl font-bold bg-gradient-to-r from-black to-gray-700 dark:from-white dark:to-gray-300 bg-clip-text text-transparent transition-all duration-300">
            SUJBOT2
          </h1>
          <p className="text-xs text-gray-500 dark:text-gray-400 transition-colors duration-300">Legal Compliance Checker</p>
        </div>
      </div>

      <div className="flex items-center gap-4 animate-fade-in">
        <LanguageSwitcher />
        <ThemeToggle />
      </div>
    </div>
  );
};
