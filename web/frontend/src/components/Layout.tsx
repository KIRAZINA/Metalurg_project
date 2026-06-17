import { Link, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

export function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div style={{ display: "flex", minHeight: "100vh", margin: 0, fontFamily: "system-ui, sans-serif" }}>
      <nav style={{ width: 220, background: "#1a1a2e", color: "#eee", padding: "1rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        <h2 style={{ margin: "0 0 1rem", color: "#e94560" }}>⚙️ Test Metal</h2>
        {user && <p style={{ fontSize: "0.85rem", opacity: 0.7 }}>👤 {user.email}</p>}
        <Link to="/" style={linkStyle}>🏠 Dashboard</Link>
        <Link to="/datasets" style={linkStyle}>📊 Datasets</Link>
        <Link to="/optimizations" style={linkStyle}>🎯 Optimizations</Link>
        <Link to="/tasks" style={linkStyle}>📋 Tasks</Link>
        <div style={{ flex: 1 }} />
        {user && (
          <button onClick={handleLogout} style={{ ...linkStyle, background: "none", border: "none", cursor: "pointer", textAlign: "left", color: "#e94560" }}>
            🚪 Logout
          </button>
        )}
      </nav>
      <main style={{ flex: 1, padding: "2rem", background: "#f5f5f5", overflowY: "auto" }}>
        <Outlet />
      </main>
    </div>
  );
}

const linkStyle: React.CSSProperties = {
  color: "#ddd",
  textDecoration: "none",
  padding: "0.5rem 0.75rem",
  borderRadius: 6,
  fontSize: "0.95rem",
};
