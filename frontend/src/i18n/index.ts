/**
 * i18n Configuration - Internationalization setup using react-i18next
 *
 * Features:
 * - Auto-detect browser language on first visit
 * - Persist language selection in localStorage
 * - Fallback to Czech if detection fails
 */

import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import cs from './locales/cs.json';
import en from './locales/en.json';

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      cs: { translation: cs },
      en: { translation: en },
    },
    fallbackLng: 'cs',
    supportedLngs: ['cs', 'en'],
    interpolation: {
      escapeValue: false, // React already escapes values
    },
    detection: {
      // Detection order: localStorage first, then browser language
      order: ['localStorage', 'navigator'],
      // Cache user selection in localStorage
      caches: ['localStorage'],
      // localStorage key name
      lookupLocalStorage: 'sujbot2-language',
    },
  });

export default i18n;
