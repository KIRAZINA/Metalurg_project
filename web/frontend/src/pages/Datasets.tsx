import { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { listDatasets, uploadDataset, deleteDataset } from "../api/endpoints";
import type { Dataset } from "../types";

export default function Datasets() {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);

  const load = () => {
    setLoading(true);
    listDatasets().then(d => { setDatasets(d.items); setTotal(d.total); }).finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const handleUpload = async () => {
    const file = fileRef.current?.files?.[0];
    const name = nameRef.current?.value;
    if (!file || !name) return;
    setUploading(true);
    try {
      await uploadDataset(file, name);
      load();
      if (fileRef.current) fileRef.current.value = "";
      if (nameRef.current) nameRef.current.value = "";
    } catch (err) {
      alert(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this dataset?")) return;
    try {
      await deleteDataset(id);
      load();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Delete failed");
    }
  };

  return (
    <div>
      <h1>📊 Datasets</h1>

      <section style={{ background: "#fff", padding: "1.5rem", borderRadius: 12, marginBottom: "1.5rem" }}>
        <h2>Upload New</h2>
        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}>
          <input ref={nameRef} type="text" placeholder="Dataset name" style={inputStyle} />
          <input ref={fileRef} type="file" accept=".xls,.xlsx" style={inputStyle} />
          <button onClick={handleUpload} disabled={uploading} style={btnStyle}>
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </div>
      </section>

      <section style={{ background: "#fff", padding: "1.5rem", borderRadius: 12 }}>
        <h2>Your Datasets ({total})</h2>
        {loading ? (
          <p>Loading...</p>
        ) : datasets.length === 0 ? (
          <p>No datasets yet.</p>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ textAlign: "left", borderBottom: "2px solid #eee" }}>
                <th style={thStyle}>Name</th>
                <th style={thStyle}>Status</th>
                <th style={thStyle}>Rows</th>
                <th style={thStyle}>Size</th>
                <th style={thStyle}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map(ds => (
                <tr key={ds.id} style={{ borderBottom: "1px solid #eee" }}>
                  <td style={tdStyle}>{ds.name}</td>
                  <td style={tdStyle}><StatusBadge status={ds.status} /></td>
                  <td style={tdStyle}>{ds.row_count ?? "—"}</td>
                  <td style={tdStyle}>{(ds.file_size_bytes / 1024).toFixed(1)} KB</td>
                  <td style={{ ...tdStyle, display: "flex", gap: "0.5rem" }}>
                    <button onClick={() => navigate(`/datasets/${ds.id}`)} style={smallBtn}>Open</button>
                    <button onClick={() => handleDelete(ds.id)} style={{ ...smallBtn, background: "#e74c3c" }}>Delete</button>
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

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = { uploading: "#f39c12", processing: "#3498db", ready: "#2ecc71", failed: "#e74c3c" };
  return <span style={{ background: colors[status] || "#95a5a6", color: "#fff", padding: "2px 8px", borderRadius: 4, fontSize: "0.8rem" }}>{status}</span>;
}

const inputStyle: React.CSSProperties = { padding: "0.5rem", borderRadius: 6, border: "1px solid #ccc", fontSize: "0.9rem" };
const btnStyle: React.CSSProperties = { padding: "0.5rem 1rem", borderRadius: 6, border: "none", background: "#4361ee", color: "#fff", cursor: "pointer", fontSize: "0.9rem" };
const smallBtn: React.CSSProperties = { ...btnStyle, padding: "0.3rem 0.75rem", fontSize: "0.8rem" };
const thStyle: React.CSSProperties = { padding: "0.75rem 0.5rem", fontSize: "0.85rem", color: "#666" };
const tdStyle: React.CSSProperties = { padding: "0.75rem 0.5rem", fontSize: "0.9rem" };
