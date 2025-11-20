/**
 * Authentication Context - Manages user authentication state
 *
 * Cookie-based authentication with httpOnly JWT tokens
 * - JWT tokens stored in httpOnly cookies (XSS protection)
 * - No localStorage usage (prevents token theft via XSS)
 * - Backend manages session (stateless JWT validation)
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { apiService, type UserProfile } from '../services/api';

interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: UserProfile | null;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [user, setUser] = useState<UserProfile | null>(null);

  // Check for existing session on mount (validates httpOnly cookie)
  useEffect(() => {
    const verifySession = async () => {
      try {
        // Verify JWT cookie with backend (/auth/me endpoint)
        // Cookie is sent automatically by browser (httpOnly)
        const userProfile = await apiService.getCurrentUser();

        // Session is valid
        setUser(userProfile);
        setIsAuthenticated(true);
      } catch (error) {
        // No valid session (cookie expired, invalid, or missing)
        console.info('No existing session found');
        setUser(null);
        setIsAuthenticated(false);
      } finally {
        setIsLoading(false);
      }
    };

    verifySession();
  }, []);

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      // Call backend login endpoint
      // Backend sets httpOnly cookie in response
      const response = await apiService.login(email, password);

      // Update state with user profile
      setUser(response.user);
      setIsAuthenticated(true);
      return true;
    } catch (error) {
      console.error('Login failed:', error);
      setUser(null);
      setIsAuthenticated(false);
      return false;
    }
  };

  const logout = async () => {
    try {
      // Call backend logout endpoint (clears httpOnly cookie)
      await apiService.logout();
    } catch (error) {
      console.error('Logout failed:', error);
      // Continue anyway to clear local state
    } finally {
      // Update local state
      setUser(null);
      setIsAuthenticated(false);
    }
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, isLoading, user, login, logout }}>
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
