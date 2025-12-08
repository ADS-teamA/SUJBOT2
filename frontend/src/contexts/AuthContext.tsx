/**
 * Authentication Context - Manages user authentication state
 *
 * Cookie-based authentication with httpOnly JWT tokens
 * - JWT tokens stored in httpOnly cookies (XSS protection)
 * - No localStorage usage (prevents token theft via XSS)
 * - Backend manages session (stateless JWT validation)
 */

import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import { apiService, type UserProfile } from '../services/api';

interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: UserProfile | null;
  error: string | null;
  login: (email: string, password: string) => Promise<{success: boolean, error?: string}>;
  logout: () => Promise<void>;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [user, setUser] = useState<UserProfile | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Check for existing session on mount (validates httpOnly cookie)
  useEffect(() => {
    // AbortController for cleanup when component unmounts during fetch
    const abortController = new AbortController();

    const verifySession = async () => {
      try {
        // Verify JWT cookie with backend (/auth/me endpoint)
        // Cookie is sent automatically by browser (httpOnly)
        const userProfile = await apiService.getCurrentUser(abortController.signal);

        // Don't update state if component was unmounted
        if (abortController.signal.aborted) return;

        // Session is valid
        setUser(userProfile);
        setIsAuthenticated(true);
      } catch (error) {
        // Don't update state if this was an abort (component unmounted)
        if (error instanceof Error && error.name === 'AbortError') {
          return;
        }
        // No valid session (cookie expired, invalid, or missing)
        console.info('No existing session found');
        setUser(null);
        setIsAuthenticated(false);
      } finally {
        // Don't update loading state if aborted
        if (!abortController.signal.aborted) {
          setIsLoading(false);
        }
      }
    };

    verifySession();

    // Cleanup: abort pending request when component unmounts
    return () => {
      abortController.abort();
    };
  }, []);

  const login = async (email: string, password: string): Promise<{success: boolean, error?: string}> => {
    setError(null); // Clear previous errors
    try {
      // Call backend login endpoint
      // Backend sets httpOnly cookie in response
      const response = await apiService.login(email, password);

      // Update state with user profile
      setUser(response.user);
      setIsAuthenticated(true);
      return { success: true };
    } catch (error) {
      // Don't show error if request was aborted (e.g., user navigated away)
      if (error instanceof Error && error.name === 'AbortError') {
        return { success: false, error: 'Request cancelled' };
      }

      console.error('Login failed:', error);
      setUser(null);
      setIsAuthenticated(false);

      // Surface actionable error message to user
      const errorMessage = error instanceof Error ? error.message : 'Login failed';
      setError(errorMessage);
      return { success: false, error: errorMessage };
    }
  };

  const logout = async () => {
    setError(null); // Clear any auth errors
    try {
      // Call backend logout endpoint (clears httpOnly cookie)
      await apiService.logout();
    } catch (error) {
      console.error('Logout failed:', error);
      // Continue anyway to clear local state (graceful degradation)
      // Don't surface error to user - logout should always succeed locally
    } finally {
      // Update local state
      setUser(null);
      setIsAuthenticated(false);
    }
  };

  const clearError = () => {
    setError(null);
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, isLoading, user, error, login, logout, clearError }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
