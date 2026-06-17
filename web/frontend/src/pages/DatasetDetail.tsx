import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getDataset, updateDataset, getDatasetRegressions, triggerRegression } from "../api/endpoints";
import type { Dataset, RegressionModel } from "../types";

export default function DatasetDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [ds, setDs] = useState<Dataset | null>(null);
  const [regressions, setRegressions] = useState<RegressionModel[]>([]);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (!id) return;
    getDataset(id).then(d => { setDs(d); setName(d.name); setDescription(d.description || ""); });
    getDatasetRegressions(id).then(r => setRegressions(r.items)).catch(() => {});
    setLoading(false);
  }, [id]);

  const handleUpdate = async () => {
    if (!id) return;
    try { const d = await updateDataset(id, { name, description: description || undefined }); setDs(d); } catch (err) { alert(err); }
  };

  const handleRunRegression = async () => {
    if (!id) return;
    setRunning(true);
    try {
      const result = await triggerRegression(id);
      alert(`Task started: ${result.task_id}`);
    } catch (err) { alert(err); } finally { setRunning(false); }
  };

  if (loading || !ds) return <p>Loading...</p>;

  return (
    <div>
      <button onClick={() => navigate("/datasets")} style={backBtn}>← Back</button>
      <h1>📊 {ds.name}</h1>

      <div style={{ display: "flex", gap: "1rem", marginBottom: "1.5rem" }}>
        <MetricCard label="Status" value={ds.status} />
        <MetricCard label="Rows" value={ds.row_count?.toString() || "—"} />
        <MetricCard label="Columns" value={ds.column_count?.toString() || "—"} />
        <MetricCard label="Size" value={`${(ds.file_size_bytes / 1024).toFixed(1)} KB`} />
      </div>

      <section style={{ background: "#fff", padding: "1.5rem", borderRadius: 12, marginBottom: "1.5rem" }}>
        <h2>Details</h2>
        <p><strong>File:</strong> {ds.original_filename}</p>
        {ds.error_message && <p style={{ color: "red" }}>{ds.error_message}</p>}
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
          <input value={name} onChange={e => setName(e.target.value)} style={inputStyle} />
          <input value={description} onChange={e => setDescription(e.target.value)} placeholder="Description" style={inputStyle} />
          <button onClick={handleUpdate} style={btnStyle}>Update</button>
        </div>
      </section>

      <section style={{ background: "#fff", padding: "1.5rem", borderRadius: 12, marginBottom: "1.5rem" }}>
        <h2>🔬 Regressions</h2>
        <button onClick={handleRunRegression} disabled={running} style={btnStyle}>
          {running ? "Running..." : "Run Regression"}
        </button>

        {regressions.length === 0 ? (
          <p style={{ marginTop: "1rem" }}>No regressions yet.</p>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "1rem" }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #eee", textAlign: "left" }}>
                <th style={thStyle}>X → Y</th>
                <th style={thStyle}>R²</th>
                <th style={thStyle}>Slope</th>
                <th style={thStyle}>Intercept</th>
                <th style={thStyle}>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {regressions.map(r => (
                <tr key={r.id} style={{ borderBottom: "1px solid #eee" }}>
                  <td style={tdStyle}>{r.x_column} → {r.y_column}</td>
                  <td style={tdStyle}>{r.r_squared.toFixed(4)}</td>
                  <td style={tdStyle}>{r.slope.toFixed(6)}</td>
                  <td style={tdStyle}>{r.intercept.toFixed(6)}</td>
                  <td style={tdStyle}>{r.confidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ flex: 1, padding: "1rem", background: "#fff", borderRadius: 8, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
      <p style={{ margin: 0, fontSize: "0.8rem", color: "#666" }}>{label}</p>
      <p style={{ margin: "0.25rem 0 0", fontSize: "1.25rem", fontWeight: 600 }}>{value}</p>
    </div>
  );
}

const backBtn: React.CSSProperties = { background: "none", border: "1px solid #ccc", borderRadius: 6, padding: "0.4rem 0.75rem", cursor: "pointer", marginBottom: "1rem" };
const inputStyle: React.CSSProperties = { padding: "0.5rem", borderRadius: 6, border: "1px solid #ccc", fontSize: "0.9rem", flex: 1, minWidth: 200 };
const btnStyle: React.CSSProperties = { padding: "0.5rem 1rem", borderRadius: 6, border: "none", background: "#4361ee", color: "#fff", cursor: "pointer" };
const thStyle: React.CSSProperties = { padding: "0.5rem", fontSize: "0.85rem", color: "#666" };
const tdStyle: React.CSSProperties = { padding: "0.5rem", fontSize: "0.9rem" };
