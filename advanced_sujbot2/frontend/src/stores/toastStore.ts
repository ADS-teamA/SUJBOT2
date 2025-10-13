import { create } from 'zustand';
import i18n from '@/i18n';

export type ToastVariant = 'default' | 'success' | 'error' | 'warning' | 'info';

export interface Toast {
  id: string;
  title: string;
  description?: string;
  variant: ToastVariant;
  duration?: number;
}

interface ToastStore {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  success: (titleKey: string, descriptionOrParams?: string | object) => void;
  error: (titleKey: string, descriptionOrParams?: string | object) => void;
  warning: (titleKey: string, descriptionOrParams?: string | object) => void;
  info: (titleKey: string, descriptionOrParams?: string | object) => void;
}

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],

  addToast: (toast) => {
    const id = Math.random().toString(36).substring(2, 9);
    const newToast = { ...toast, id };

    set((state) => ({
      toasts: [...state.toasts, newToast],
    }));

    // Auto-remove toast after duration (default 5000ms)
    const duration = toast.duration ?? 5000;
    if (duration > 0) {
      setTimeout(() => {
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        }));
      }, duration);
    }
  },

  removeToast: (id) => {
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    }));
  },

  success: (titleKey, descriptionOrParams) => {
    const title = typeof descriptionOrParams === 'object'
      ? i18n.t(titleKey, descriptionOrParams)
      : i18n.t(titleKey);

    const description = typeof descriptionOrParams === 'string'
      ? descriptionOrParams
      : undefined;

    useToastStore.getState().addToast({
      title,
      description,
      variant: 'success',
    });
  },

  error: (titleKey, descriptionOrParams) => {
    const title = typeof descriptionOrParams === 'object'
      ? i18n.t(titleKey, descriptionOrParams)
      : i18n.t(titleKey);

    const description = typeof descriptionOrParams === 'string'
      ? descriptionOrParams
      : undefined;

    useToastStore.getState().addToast({
      title,
      description,
      variant: 'error',
      duration: 7000, // Errors stay longer
    });
  },

  warning: (titleKey, descriptionOrParams) => {
    const title = typeof descriptionOrParams === 'object'
      ? i18n.t(titleKey, descriptionOrParams)
      : i18n.t(titleKey);

    const description = typeof descriptionOrParams === 'string'
      ? descriptionOrParams
      : undefined;

    useToastStore.getState().addToast({
      title,
      description,
      variant: 'warning',
    });
  },

  info: (titleKey, descriptionOrParams) => {
    const title = typeof descriptionOrParams === 'object'
      ? i18n.t(titleKey, descriptionOrParams)
      : i18n.t(titleKey);

    const description = typeof descriptionOrParams === 'string'
      ? descriptionOrParams
      : undefined;

    useToastStore.getState().addToast({
      title,
      description,
      variant: 'info',
    });
  },
}));
