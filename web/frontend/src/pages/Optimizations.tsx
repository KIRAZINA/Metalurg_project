import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listOptimizations, listDatasets, createOptimization, deleteOptimization } from "../api/endpoints";
import type { ParetoOptimization, Dataset } from "../types";

export default function Optimizations() {
  const navigate = useNavigate();
  const [items, setItems] = useState<ParetoOptimization[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);

  const [dsList, setDsList] = useState<Dataset[]>([]);
  const [selectedDs, setSelectedDs] = useState("");
  const [optName, setOptName] = useState("");
  const [targetsStr, setTargetsStr] = useState('{\n  "Fe": {"x_column": "X1", "target_value": 90},\n  "C": {"x_column": "X2", "target_value": 0.5}\n}');
  const [nPoints, setNPoints] = useState(100);

  const load = () => {
    setLoading(true);
    listOptimizations().then(d => { setItems(d.items); setTotal(d.total); }).finally(() => setLoading(false));
  };

  useEffect(() => { load(); listDatasets(1, 100).then(d => setDsList(d.items.filter(ds => ds.status === "ready"))); }, []);

  const handleCreate = async () => {
    if (!selectedDs) return alert("Select a dataset");
    setCreating(true);
    try {
      const result = await createOptimization({ dataset_id: selectedDs, name: optName || undefined, targets: targetsStr, n_points: nPoints });
      navigate(`/optimizations/${result.id}`);
    } catch (err) { alert(err); } finally { setCreating(false); }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this optimization?")) return;
    try { await deleteOptimization(id); load(); } catch (err) { alert(err); }
  };

  return (
    <div>
      <h1>🎯 Optimizations</h1>

      <section style={{ background: "#fff", padding: "1.5rem", borderRadius: 12, marginBottom: "1.5rem" }}>
        <h2>Create New</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          <select value={selectedDs} onChange={e => setSelectedDs(e.target.value)} style={inputStyle}>
            <option value="">— Select dataset —</option>
            {dsList.map(ds => <option key={ds.id} value={ds.id}>{ds.name}</option>)}
          </select>
          <input value={optName} onChange={e => setOptName(e.target.value)} placeholder="Optimization name (optional)" style={inputStyle} />
          <label>Number of points: {nPoints}</label>
          <input type="range" min={10} max={500} value={nPoints} onChange={e => setNPoints(Number(e.target.value))} />
          <textarea value={targetsStr} onChange={e => setTargetsStr(e.target.value)} rows={6} style={{ ...inputStyle, fontFamily: "monospace" }} />
          <button onClick={handleCreate} disabled={creating} style={btnStyle}>{creating ? "Creating..." : "Create & Run"}</button>
        </div>
      </section>

      <section style={{ background: "#fff", padding: "1.5rem", borderRadius: 12 }}>
        <h2>Your Optimizations ({total})</h2>
        {loading ? <p>Loading...</p> : items.length === 0 ? <p>None yet.</p> : (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #eee", textAlign: "left" }}>
                <th style={thStyle}>Name</th>
                <th style={thStyle}>Status</th>
                <th style={thStyle}>Points</th>
                <th style={thStyle}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map(o => (
                <tr key={o.id} style={{ borderBottom: "1px solid #eee" }}>
                  <td style={tdStyle}>{o.name || "Unnamed"}</td>
                  <td style={tdStyle}>{o.status}</td>
                  <td style={tdStyle}>{o.n_points}</td>
                  <td style={tdStyle}>
                    <button onClick={() => navigate(`/optimizations/${o.id}`)} style={smallBtn}>Open</button>
                    <button onClick={() => handleDelete(o.id)} style={{ ...smallBtn, background: "#e74c3c", marginLeft: "0.5rem" }}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

const inputStyle: React.CSSProperties = { padding: "0.5rem", borderRadius: 6, border: "1px solid #ccc", fontSize: "0.9rem" };
const btnStyle: React.CSSProperties = { padding: "0.5rem 1rem", borderRadius: 6, border: "none", background: "#4361ee", color: "#fff", cursor: "pointer" };
const smallBtn: React.CSSProperties = { ...btnStyle, padding: "0.3rem 0.75rem", fontSize: "0.8rem" };
const thStyle: React.CSSProperties = { padding: "0.5rem", fontSize: "0.85rem", color: "#666" };
const tdStyle: React.CSSProperties = { padding: "0.5rem", fontSize: "0.9rem" };
