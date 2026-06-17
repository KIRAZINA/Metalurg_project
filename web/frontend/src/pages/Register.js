import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { register } from "../api/endpoints";
export default function Register() {
    const navigate = useNavigate();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [fullName, setFullName] = useState("");
    const [error, setError] = useState("");
    const [success, setSuccess] = useState("");
    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        setSuccess("");
        try {
            await register(email, password, fullName || undefined);
            setSuccess("Registration successful! Redirecting to login...");
            setTimeout(() => navigate("/login"), 1500);
        }
        catch (err) {
            setError(err instanceof Error ? err.message : "Registration failed");
        }
    };
    return (_jsxs("div", { style: { maxWidth: 400, margin: "4rem auto", padding: "2rem", background: "#fff", borderRadius: 12, boxShadow: "0 2px 12px rgba(0,0,0,0.08)" }, children: [_jsx("h1", { style: { textAlign: "center", marginBottom: "1.5rem" }, children: "Register" }), error && _jsx("p", { style: { color: "red", background: "#fee", padding: "0.5rem", borderRadius: 6 }, children: error }), success && _jsx("p", { style: { color: "green", background: "#efe", padding: "0.5rem", borderRadius: 6 }, children: success }), _jsxs("form", { onSubmit: handleSubmit, style: { display: "flex", flexDirection: "column", gap: "1rem" }, children: [_jsx("input", { type: "text", placeholder: "Full Name (optional)", value: fullName, onChange: e => setFullName(e.target.value), style: inputStyle }), _jsx("input", { type: "email", placeholder: "Email", value: email, onChange: e => setEmail(e.target.value), required: true, style: inputStyle }), _jsx("input", { type: "password", placeholder: "Password (min 8 chars)", value: password, onChange: e => setPassword(e.target.value), required: true, minLength: 8, style: inputStyle }), _jsx("button", { type: "submit", style: btnStyle, children: "Register" })] }), _jsxs("p", { style: { textAlign: "center", marginTop: "1rem" }, children: ["Already have an account? ", _jsx(Link, { to: "/login", children: "Sign in" })] })] }));
}
const inputStyle = {
    padding: "0.75rem", borderRadius: 8, border: "1px solid #ccc", fontSize: "1rem",
};
const btnStyle = {
    padding: "0.75rem", borderRadius: 8, border: "none", background: "#1a1a2e", color: "#fff", fontSize: "1rem", cursor: "pointer",
};
