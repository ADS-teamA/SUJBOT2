/**
 * LoginPage Component - Authentication gateway for SUJBOT2
 *
 * Design: Sophisticated, centered layout matching the SUJBOT2 aesthetic
 * - Cormorant Garamond serif typography for branding
 * - Monochromatic palette with subtle gradients
 * - Smooth 750ms transitions matching theme system
 * - Ceremonial entrance feel for professional legal/technical system
 */

import React, { useState } from 'react';
import { Lock, Mail, AlertCircle } from 'lucide-react';
import { cn } from '../design-system/utils/cn';
import { useAuth } from '../contexts/AuthContext';
import './LoginPage.css';

export function LoginPage() {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const result = await login(email, password);
      if (!result.success) {
        // Use specific error message from AuthContext (e.g., "Invalid credentials", "Failed to connect to backend")
        setError(result.error || 'Login failed. Please try again.');
      }
    } catch (err) {
      // Fallback for unexpected errors (shouldn't happen with new error handling)
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={cn(
      'min-h-screen flex items-center justify-center',
      'bg-white dark:bg-accent-950',
      'px-6 py-12',
      'relative overflow-hidden'
    )}>
      {/* Gradient background effects (matching WelcomeScreen) */}
      <div
        className="absolute inset-0 -z-10"
        style={{
          background: 'var(--gradient-mesh-light)',
        }}
      />
      <div
        className="absolute inset-0 -z-10 dark:block hidden"
        style={{
          background: 'var(--gradient-mesh-dark)',
        }}
      />
      <div
        className="absolute inset-0 -z-10"
        style={{
          background: 'var(--gradient-light)',
        }}
      />
      <div
        className="absolute inset-0 -z-10 dark:block hidden"
        style={{
          background: 'var(--gradient-dark)',
        }}
      />

      {/* Floating particles effect (subtle decoration) */}
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className={cn(
              'absolute rounded-full',
              'bg-accent-200/20 dark:bg-accent-700/10'
            )}
            style={{
              width: `${Math.random() * 150 + 50}px`,
              height: `${Math.random() * 150 + 50}px`,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animation: `float ${Math.random() * 20 + 15}s ease-in-out infinite`,
              animationDelay: `${Math.random() * 5}s`,
            }}
          />
        ))}
      </div>

      {/* Login card */}
      <div
        className="w-full max-w-md"
        style={{
          animation: 'fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        {/* Logo and branding */}
        <div className="text-center mb-8">
          {/* SUJBOT Logo */}
          <div className="flex justify-center mb-4">
            <svg
              width="80"
              height="80"
              viewBox="0 0 512 512"
              xmlns="http://www.w3.org/2000/svg"
              className={cn(
                'text-accent-900 dark:text-accent-100',
                'opacity-90'
              )}
              style={{
                filter: 'drop-shadow(0 2px 8px rgba(0, 0, 0, 0.1))',
              }}
            >
              {/* Atom + Paragraph Symbol */}
              <g transform="translate(256 256)" stroke="currentColor" fill="none" strokeLinecap="round">
                {/* Orbitals */}
                <ellipse rx="185" ry="110" strokeWidth="16" />
                <ellipse rx="185" ry="110" strokeWidth="16" transform="rotate(60)" />
                <ellipse rx="185" ry="110" strokeWidth="16" transform="rotate(-60)" />

                {/* Electrons */}
                <circle r="20" cx="185" cy="0" fill="currentColor" stroke="none" />
                <circle r="20" cx="-92.5" cy="160" fill="currentColor" stroke="none" />
                <circle r="20" cx="-92.5" cy="-160" fill="currentColor" stroke="none" />

                {/* § symbol */}
                <text
                  x="0"
                  y="35"
                  fontSize="140"
                  fontWeight="bold"
                  fill="currentColor"
                  textAnchor="middle"
                  fontFamily="serif"
                >§</text>
              </g>
            </svg>
          </div>

          {/* App name */}
          <h1
            className={cn(
              'text-5xl font-light tracking-tight mb-2',
              'text-accent-950 dark:text-accent-50'
            )}
            style={{ fontFamily: 'var(--font-display)' }}
          >
            SUJBOT2
          </h1>

          {/* Tagline */}
          <p className={cn(
            'text-sm font-light',
            'text-accent-500 dark:text-accent-400'
          )}>
            Legal & Technical Document Intelligence
          </p>
        </div>

        {/* Login form */}
        <div className={cn(
          'p-8 rounded-2xl',
          'bg-white/80 dark:bg-accent-900/50',
          'backdrop-blur-sm',
          'border border-accent-200 dark:border-accent-800',
          'shadow-xl'
        )}>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email field */}
            <div>
              <label
                htmlFor="email"
                className={cn(
                  'block text-sm font-medium mb-2',
                  'text-accent-700 dark:text-accent-300'
                )}
              >
                Email
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail size={18} className="text-accent-400 dark:text-accent-500" />
                </div>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className={cn(
                    'w-full pl-10 pr-4 py-3 rounded-lg',
                    'bg-white dark:bg-accent-800',
                    'border border-accent-300 dark:border-accent-700',
                    'text-accent-900 dark:text-accent-100',
                    'placeholder-accent-400 dark:placeholder-accent-500',
                    'focus:outline-none focus:ring-2 focus:ring-accent-500 dark:focus:ring-accent-400',
                    'focus:border-transparent',
                    'transition-all duration-200'
                  )}
                  placeholder="admin@sujbot.local"
                  required
                  autoComplete="email"
                />
              </div>
            </div>

            {/* Password field */}
            <div>
              <label
                htmlFor="password"
                className={cn(
                  'block text-sm font-medium mb-2',
                  'text-accent-700 dark:text-accent-300'
                )}
              >
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock size={18} className="text-accent-400 dark:text-accent-500" />
                </div>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className={cn(
                    'w-full pl-10 pr-4 py-3 rounded-lg',
                    'bg-white dark:bg-accent-800',
                    'border border-accent-300 dark:border-accent-700',
                    'text-accent-900 dark:text-accent-100',
                    'placeholder-accent-400 dark:placeholder-accent-500',
                    'focus:outline-none focus:ring-2 focus:ring-accent-500 dark:focus:ring-accent-400',
                    'focus:border-transparent',
                    'transition-all duration-200'
                  )}
                  placeholder="Enter password"
                  required
                  autoComplete="current-password"
                />
              </div>
            </div>

            {/* Error message */}
            {error && (
              <div
                className={cn(
                  'p-3 rounded-lg flex items-start gap-2',
                  'bg-red-50 dark:bg-red-900/20',
                  'border border-red-200 dark:border-red-800',
                  'text-red-800 dark:text-red-200',
                  'text-sm'
                )}
                style={{
                  animation: 'fadeIn 0.3s ease-in',
                }}
              >
                <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}

            {/* Submit button */}
            <button
              type="submit"
              disabled={isLoading}
              className={cn(
                'w-full py-3 px-4 rounded-lg',
                'bg-accent-900 dark:bg-accent-100',
                'text-white dark:text-accent-900',
                'font-medium',
                'hover:bg-accent-800 dark:hover:bg-accent-200',
                'focus:outline-none focus:ring-2 focus:ring-accent-500 dark:focus:ring-accent-400',
                'focus:ring-offset-2',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'transition-all duration-200',
                'shadow-lg hover:shadow-xl'
              )}
            >
              {isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg
                    className="animate-spin h-5 w-5"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Authenticating...
                </span>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          {/* Footer hint */}
          <div className="mt-6 text-center">
            <p className={cn(
              'text-xs',
              'text-accent-400 dark:text-accent-600'
            )}>
              Authorized Access Only
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}
