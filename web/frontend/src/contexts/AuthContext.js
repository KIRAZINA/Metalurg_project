import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext, useState, useCallback } from "react";
import { setTokens, clearTokens, setLogoutHandler } from "../api/client";
const AuthContext = createContext(null);
export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const logout = useCallback(() => {
        clearTokens();
        setUser(null);
    }, []);
    const loginFn = useCallback((tokens, user) => {
        setTokens(tokens.access_token, tokens.refresh_token);
        setUser(user);
    }, []);
    setLogoutHandler(logout);
    return (_jsx(AuthContext.Provider, { value: { user, isAuthenticated: !!user, login: loginFn, logout }, children: children }));
}
export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx)
        throw new Error("useAuth must be used within AuthProvider");
    return ctx;
}
