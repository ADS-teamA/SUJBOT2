/**
 * i18next Type Declarations - Provides TypeScript support for translations
 *
 * This module augmentation:
 * - Provides autocomplete for known translation keys
 * - Allows dynamic string keys for patterns like PHASE_MAP lookups
 * - Ensures t() returns string (never null)
 *
 * Note: We intentionally do NOT enable strict key checking because this codebase
 * uses dynamic key patterns (e.g., t(config.nameKey) where keys come from objects).
 * Strict typing would require type assertions everywhere, reducing code clarity.
 *
 * Based on: https://www.i18next.com/overview/typescript
 */

import 'i18next';
import type cs from './locales/cs.json';

// Flatten nested object type to dot-notation keys
type FlattenKeys<T, Prefix extends string = ''> = T extends object
  ? {
      [K in keyof T]: K extends string
        ? T[K] extends object
          ? FlattenKeys<T[K], `${Prefix}${K}.`>
          : `${Prefix}${K}`
        : never;
    }[keyof T]
  : never;

// All valid translation keys in dot notation
export type TranslationKey = FlattenKeys<typeof cs>;

declare module 'i18next' {
  interface CustomTypeOptions {
    defaultNS: 'translation';
    // Explicitly allow any string to support dynamic keys
    // while still providing autocomplete via TranslationKey type
    returnNull: false;
  }
}

declare module 'react-i18next' {
  interface CustomTypeOptions {
    defaultNS: 'translation';
    returnNull: false;
  }
}
