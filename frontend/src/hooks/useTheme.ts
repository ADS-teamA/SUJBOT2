/**
 * useTheme hook - Always light mode
 */

import { useEffect } from 'react';

export type Theme = 'light';

export function useTheme() {
  const theme: Theme = 'light';

  useEffect(() => {
    // Apply light theme to document
    const root = window.document.documentElement;
    root.classList.remove('dark');
    root.classList.add('light');
  }, []);

  return { theme };
}
