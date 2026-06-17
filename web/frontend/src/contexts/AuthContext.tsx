import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
import { setTokens, clearTokens, setLogoutHandler } from "../api/client";
import type { User, AuthTokens } from "../types";

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (tokens: AuthTokens, user: User) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  const logout = useCallback(() => {
    clearTokens();
    setUser(null);
  }, []);

  const loginFn = useCallback((tokens: AuthTokens, user: User) => {
    setTokens(tokens.access_token, tokens.refresh_token);
    setUser(user);
  }, []);

  setLogoutHandler(logout);

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: !!user, login: loginFn, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
