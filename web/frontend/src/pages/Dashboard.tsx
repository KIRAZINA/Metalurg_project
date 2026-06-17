import { useEffect, useState } from "react";
import { useAuth } from "../contexts/AuthContext";
import { listDatasets, listOptimizations, listTasks } from "../api/endpoints";

export default function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState({ datasets: 0, optimizations: 0, tasks: 0 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      listDatasets(1, 1).then(d => d.total),
      listOptimizations(1, 1).then(o => o.total),
      listTasks(1, 1).then(t => t.total),
    ])
      .then(([datasets, optimizations, tasks]) => setStats({ datasets, optimizations, tasks }))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <h1>Dashboard</h1>
      <p>Welcome, {user?.full_name || user?.email}!</p>

      {loading ? (
        <p>Loading stats...</p>
      ) : (
        <div style={{ display: "flex", gap: "1rem", marginTop: "1.5rem" }}>
          <StatCard label="Datasets" value={stats.datasets} color="#4361ee" />
          <StatCard label="Optimizations" value={stats.optimizations} color="#f72585" />
          <StatCard label="Tasks" value={stats.tasks} color="#4cc9f0" />
        </div>
      )}

      <div style={{ marginTop: "2rem", lineHeight: 2 }}>
        <h2>Quick Actions</h2>
        <ul>
          <li><a href="/datasets">📊 Browse Datasets</a></li>
          <li><a href="/optimizations">🎯 Create Optimization</a></li>
          <li><a href="/tasks">📋 View Tasks</a></li>
        </ul>
      </div>
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ flex: 1, padding: "1.5rem", background: "#fff", borderRadius: 12, boxShadow: "0 2px 8px rgba(0,0,0,0.06)", borderTop: `4px solid ${color}` }}>
      <h3 style={{ margin: 0, fontSize: "0.9rem", color: "#666" }}>{label}</h3>
      <p style={{ margin: "0.5rem 0 0", fontSize: "2rem", fontWeight: 700 }}>{value}</p>
    </div>
  );
}
