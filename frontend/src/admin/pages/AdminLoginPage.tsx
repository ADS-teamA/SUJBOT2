/**
 * Admin Login Page
 *
 * Styled similarly to the main LoginPage but for admin access.
 * Redirects to /admin on successful login.
 */

import { useState, type FormEvent } from 'react';
import { Mail, Lock, AlertCircle, Shield } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL !== undefined
  ? import.meta.env.VITE_API_BASE_URL
  : 'http://localhost:8000';

export function AdminLoginPage() {
  const { t } = useTranslation();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/admin/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({ detail: 'Login failed' }));
        throw new Error(data.detail || 'Invalid credentials');
      }

      // Redirect to admin dashboard
      window.location.href = '/admin';
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={cn(
      'min-h-screen flex items-center justify-center',
      'bg-gradient-to-br from-accent-100 via-accent-50 to-white',
      'dark:from-accent-950 dark:via-accent-900 dark:to-accent-950'
    )}>
      <div className={cn(
        'w-full max-w-md p-8 mx-4',
        'bg-white/80 dark:bg-accent-900/80',
        'backdrop-blur-xl',
        'rounded-2xl',
        'border border-accent-200/50 dark:border-accent-700/50',
        'shadow-2xl'
      )}>
        {/* Header */}
        <div className="text-center mb-8">
          <div className={cn(
            'inline-flex items-center justify-center',
            'w-16 h-16 mb-4',
            'bg-accent-900 dark:bg-accent-100',
            'rounded-2xl'
          )}>
            <Shield size={32} className="text-white dark:text-accent-900" />
          </div>
          <h1 className="text-2xl font-bold text-accent-900 dark:text-accent-100">
            {t('admin.login.title', 'Admin Portal')}
          </h1>
          <p className="text-accent-600 dark:text-accent-400 mt-2">
            {t('admin.login.subtitle', 'Sign in with your admin credentials')}
          </p>
        </div>

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Email Field */}
          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-accent-700 dark:text-accent-300 mb-2"
            >
              {t('login.email', 'Email')}
            </label>
            <div className="relative">
              <Mail
                size={18}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-accent-400"
              />
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
                className={cn(
                  'w-full pl-10 pr-4 py-3',
                  'bg-white dark:bg-accent-800',
                  'border border-accent-200 dark:border-accent-700',
                  'rounded-lg',
                  'text-accent-900 dark:text-accent-100',
                  'placeholder-accent-400',
                  'focus:outline-none focus:ring-2 focus:ring-accent-500',
                  'transition-all duration-200'
                )}
                placeholder={t('login.emailPlaceholder', 'admin@example.com')}
              />
            </div>
          </div>

          {/* Password Field */}
          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-accent-700 dark:text-accent-300 mb-2"
            >
              {t('login.password', 'Password')}
            </label>
            <div className="relative">
              <Lock
                size={18}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-accent-400"
              />
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                className={cn(
                  'w-full pl-10 pr-4 py-3',
                  'bg-white dark:bg-accent-800',
                  'border border-accent-200 dark:border-accent-700',
                  'rounded-lg',
                  'text-accent-900 dark:text-accent-100',
                  'placeholder-accent-400',
                  'focus:outline-none focus:ring-2 focus:ring-accent-500',
                  'transition-all duration-200'
                )}
                placeholder="••••••••"
              />
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className={cn(
              'flex items-center gap-2 p-3',
              'bg-red-50 dark:bg-red-900/20',
              'border border-red-200 dark:border-red-800',
              'rounded-lg',
              'text-red-700 dark:text-red-300'
            )}>
              <AlertCircle size={18} />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className={cn(
              'w-full py-3 px-4',
              'bg-accent-900 dark:bg-accent-100',
              'text-white dark:text-accent-900',
              'font-medium',
              'rounded-lg',
              'hover:bg-accent-800 dark:hover:bg-accent-200',
              'focus:outline-none focus:ring-2 focus:ring-accent-500 focus:ring-offset-2',
              'transition-all duration-200',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-2">
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                {t('login.signingIn', 'Signing in...')}
              </span>
            ) : (
              t('login.signIn', 'Sign In')
            )}
          </button>
        </form>

        {/* Back to main app link */}
        <div className="mt-6 text-center">
          <a
            href="/"
            className="text-sm text-accent-600 dark:text-accent-400 hover:text-accent-900 dark:hover:text-accent-100"
          >
            {t('admin.login.backToApp', '← Back to main application')}
          </a>
        </div>
      </div>
    </div>
  );
}
