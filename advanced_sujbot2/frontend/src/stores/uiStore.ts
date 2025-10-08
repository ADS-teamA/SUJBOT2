import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { Language, Theme } from '@/types';
import i18n from '@/i18n';

interface UIStore {
  language: Language;
  theme: Theme;
  sidebarCollapsed: boolean;

  setLanguage: (language: Language) => void;
  toggleLanguage: () => void;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
}

export const useUIStore = create<UIStore>()(
  devtools(
    persist(
      (set, get) => ({
        language: 'cs',
        theme: 'light',
        sidebarCollapsed: false,

        setLanguage: (language) => {
          set({ language });
          i18n.changeLanguage(language);
        },

        toggleLanguage: () => {
          const newLang = get().language === 'cs' ? 'en' : 'cs';
          get().setLanguage(newLang);
        },

        setTheme: (theme) => {
          set({ theme });
          if (theme === 'dark') {
            document.documentElement.classList.add('dark');
          } else {
            document.documentElement.classList.remove('dark');
          }
        },

        toggleTheme: () => {
          const newTheme = get().theme === 'light' ? 'dark' : 'light';
          get().setTheme(newTheme);
        },

        setSidebarCollapsed: (collapsed) => {
          set({ sidebarCollapsed: collapsed });
        }
      }),
      {
        name: 'ui-storage'
      }
    )
  )
);
