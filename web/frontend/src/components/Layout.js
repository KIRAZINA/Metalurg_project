import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Link, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
export function Layout() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const handleLogout = () => {
        logout();
        navigate("/login");
    };
    return (_jsxs("div", { style: { display: "flex", minHeight: "100vh", margin: 0, fontFamily: "system-ui, sans-serif" }, children: [_jsxs("nav", { style: { width: 220, background: "#1a1a2e", color: "#eee", padding: "1rem", display: "flex", flexDirection: "column", gap: "0.5rem" }, children: [_jsx("h2", { style: { margin: "0 0 1rem", color: "#e94560" }, children: "\u2699\uFE0F Test Metal" }), user && _jsxs("p", { style: { fontSize: "0.85rem", opacity: 0.7 }, children: ["\uD83D\uDC64 ", user.email] }), _jsx(Link, { to: "/", style: linkStyle, children: "\uD83C\uDFE0 Dashboard" }), _jsx(Link, { to: "/datasets", style: linkStyle, children: "\uD83D\uDCCA Datasets" }), _jsx(Link, { to: "/optimizations", style: linkStyle, children: "\uD83C\uDFAF Optimizations" }), _jsx(Link, { to: "/tasks", style: linkStyle, children: "\uD83D\uDCCB Tasks" }), _jsx("div", { style: { flex: 1 } }), user && (_jsx("button", { onClick: handleLogout, style: { ...linkStyle, background: "none", border: "none", cursor: "pointer", textAlign: "left", color: "#e94560" }, children: "\uD83D\uDEAA Logout" }))] }), _jsx("main", { style: { flex: 1, padding: "2rem", background: "#f5f5f5", overflowY: "auto" }, children: _jsx(Outlet, {}) })] }));
}
const linkStyle = {
    color: "#ddd",
    textDecoration: "none",
    padding: "0.5rem 0.75rem",
    borderRadius: 6,
    fontSize: "0.95rem",
};
