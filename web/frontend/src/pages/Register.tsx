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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    try {
      await register(email, password, fullName || undefined);
      setSuccess("Registration successful! Redirecting to login...");
      setTimeout(() => navigate("/login"), 1500);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Registration failed");
    }
  };

  return (
    <div style={{ maxWidth: 400, margin: "4rem auto", padding: "2rem", background: "#fff", borderRadius: 12, boxShadow: "0 2px 12px rgba(0,0,0,0.08)" }}>
      <h1 style={{ textAlign: "center", marginBottom: "1.5rem" }}>Register</h1>
      {error && <p style={{ color: "red", background: "#fee", padding: "0.5rem", borderRadius: 6 }}>{error}</p>}
      {success && <p style={{ color: "green", background: "#efe", padding: "0.5rem", borderRadius: 6 }}>{success}</p>}
      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        <input type="text" placeholder="Full Name (optional)" value={fullName} onChange={e => setFullName(e.target.value)} style={inputStyle} />
        <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} required style={inputStyle} />
        <input type="password" placeholder="Password (min 8 chars)" value={password} onChange={e => setPassword(e.target.value)} required minLength={8} style={inputStyle} />
        <button type="submit" style={btnStyle}>Register</button>
      </form>
      <p style={{ textAlign: "center", marginTop: "1rem" }}>
        Already have an account? <Link to="/login">Sign in</Link>
      </p>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  padding: "0.75rem", borderRadius: 8, border: "1px solid #ccc", fontSize: "1rem",
};

const btnStyle: React.CSSProperties = {
  padding: "0.75rem", borderRadius: 8, border: "none", background: "#1a1a2e", color: "#fff", fontSize: "1rem", cursor: "pointer",
};
