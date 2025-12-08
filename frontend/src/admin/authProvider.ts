/**
 * React Admin Auth Provider for SUJBOT2 Admin API
 *
 * Handles admin authentication:
 * - POST /admin/login → login
 * - POST /auth/logout → logout
 * - GET /auth/me → checkAuth, getIdentity
 */

import type { AuthProvider } from 'react-admin';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL !== undefined
  ? import.meta.env.VITE_API_BASE_URL
  : 'http://localhost:8000';

interface AdminUser {
  id: number;
  email: string;
  full_name: string | null;
  is_admin: boolean;
}

let cachedUser: AdminUser | null = null;

export const authProvider: AuthProvider = {
  login: async ({ username, password }) => {
    const response = await fetch(`${API_BASE_URL}/admin/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ email: username, password }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Login failed' }));
      throw new Error(error.detail || 'Invalid credentials');
    }

    const data = await response.json();
    cachedUser = data.user;

    return Promise.resolve();
  },

  logout: async () => {
    cachedUser = null;
    try {
      const response = await fetch(`${API_BASE_URL}/auth/logout`, {
        method: 'POST',
        credentials: 'include',
      });
      if (!response.ok) {
        console.error('Logout failed on server:', response.status);
        // Still proceed with local logout, but log the issue
      }
    } catch (error) {
      console.error('Logout request failed:', error);
      // Network error - still clear local state but log for debugging
    }
    return Promise.resolve();
  },

  checkAuth: async () => {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Not authenticated');
    }

    const user = await response.json();

    if (!user.is_admin) {
      throw new Error('Admin privileges required');
    }

    cachedUser = user;
    return Promise.resolve();
  },

  checkError: async (error) => {
    const status = error.status;
    if (status === 401 || status === 403) {
      cachedUser = null;
      throw new Error('Session expired');
    }
    // Log unexpected errors for debugging
    if (status >= 500) {
      console.error('Server error:', status, error.message || error);
    } else if (status >= 400) {
      console.warn('Client error:', status, error.message || error);
    }
    return Promise.resolve();
  },

  getIdentity: async () => {
    if (cachedUser) {
      return {
        id: cachedUser.id,
        fullName: cachedUser.full_name || cachedUser.email,
        avatar: undefined,
      };
    }

    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Not authenticated');
    }

    const user = await response.json();
    cachedUser = user;

    return {
      id: user.id,
      fullName: user.full_name || user.email,
      avatar: undefined,
    };
  },

  getPermissions: async () => {
    if (!cachedUser) {
      // Try to fetch user if not cached
      try {
        const response = await fetch(`${API_BASE_URL}/auth/me`, {
          credentials: 'include',
        });
        if (response.ok) {
          cachedUser = await response.json();
        } else {
          throw new Error('Not authenticated');
        }
      } catch {
        throw new Error('Unable to determine permissions');
      }
    }
    return cachedUser.is_admin ? 'admin' : 'user';
  },
};
