import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import csCommon from './locales/cs/common.json';
import enCommon from './locales/en/common.json';

const resources = {
  cs: {
    common: csCommon
  },
  en: {
    common: enCommon
  }
};

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: 'cs',
    fallbackLng: 'en',
    defaultNS: 'common',
    interpolation: {
      escapeValue: false
    }
  });

export default i18n;
