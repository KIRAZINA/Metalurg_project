import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { login } from "../api/endpoints";
export default function Login() {
    const { login: authLogin } = useAuth();
    const navigate = useNavigate();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        setLoading(true);
        try {
            const data = await login(email, password);
            authLogin({ access_token: data.access_token, refresh_token: data.refresh_token, token_type: data.token_type }, data.user);
            navigate("/");
        }
        catch (err) {
            setError(err instanceof Error ? err.message : "Login failed");
        }
        finally {
            setLoading(false);
        }
    };
    return (_jsxs("div", { style: { maxWidth: 400, margin: "4rem auto", padding: "2rem", background: "#fff", borderRadius: 12, boxShadow: "0 2px 12px rgba(0,0,0,0.08)" }, children: [_jsx("h1", { style: { textAlign: "center", marginBottom: "1.5rem" }, children: "Sign In" }), error && _jsx("p", { style: { color: "red", background: "#fee", padding: "0.5rem", borderRadius: 6 }, children: error }), _jsxs("form", { onSubmit: handleSubmit, style: { display: "flex", flexDirection: "column", gap: "1rem" }, children: [_jsx("input", { type: "email", placeholder: "Email", value: email, onChange: e => setEmail(e.target.value), required: true, style: inputStyle }), _jsx("input", { type: "password", placeholder: "Password", value: password, onChange: e => setPassword(e.target.value), required: true, style: inputStyle }), _jsx("button", { type: "submit", disabled: loading, style: { ...btnStyle, opacity: loading ? 0.6 : 1 }, children: loading ? "Signing in..." : "Sign In" })] }), _jsxs("p", { style: { textAlign: "center", marginTop: "1rem" }, children: ["No account? ", _jsx(Link, { to: "/register", children: "Register" })] })] }));
}
const inputStyle = {
    padding: "0.75rem",
    borderRadius: 8,
    border: "1px solid #ccc",
    fontSize: "1rem",
};
const btnStyle = {
    padding: "0.75rem",
    borderRadius: 8,
    border: "none",
    background: "#1a1a2e",
    color: "#fff",
    fontSize: "1rem",
    cursor: "pointer",
};
