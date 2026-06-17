import { useEffect, useState } from "react";
import { listTasks } from "../api/endpoints";
import type { AsyncTask } from "../types";

export default function Tasks() {
  const [tasks, setTasks] = useState<AsyncTask[]>([]);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    listTasks().then(d => setTasks(d.items)).finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const statusIcon: Record<string, string> = {
    PENDING: "🔵",
    PROGRESS: "🟡",
    SUCCESS: "🟢",
    FAILURE: "🔴",
  };

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h1>📋 Tasks</h1>
        <button onClick={load} style={btnStyle}>Refresh</button>
      </div>

      {loading ? (
        <p>Loading...</p>
      ) : tasks.length === 0 ? (
        <p>No tasks yet.</p>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          {tasks.map(t => (
            <div key={t.id} style={{ background: "#fff", padding: "1rem", borderRadius: 8, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <strong>{statusIcon[t.status] || "⚪"} {t.task_type}</strong>
                  <span style={{ marginLeft: "1rem", fontSize: "0.85rem", color: "#666" }}>{t.status}</span>
                </div>
                <span style={{ fontSize: "0.8rem", color: "#999" }}>{new Date(t.created_at).toLocaleString()}</span>
              </div>
              <div style={{ marginTop: "0.5rem", background: "#eee", borderRadius: 4, height: 8, overflow: "hidden" }}>
                <div style={{ width: `${t.progress}%`, background: t.status === "FAILURE" ? "#e74c3c" : "#4361ee", height: "100%", borderRadius: 4, transition: "width 0.3s" }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.25rem" }}>
                <span style={{ fontSize: "0.8rem", color: "#666" }}>{t.progress}%</span>
                <span style={{ fontSize: "0.75rem", color: "#999", fontFamily: "monospace" }}>ID: {t.id.slice(0, 8)}…</span>
              </div>
              {t.error_message && <p style={{ color: "red", fontSize: "0.85rem", margin: "0.5rem 0 0" }}>{t.error_message}</p>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const btnStyle: React.CSSProperties = { padding: "0.5rem 1rem", borderRadius: 6, border: "none", background: "#4361ee", color: "#fff", cursor: "pointer" };
