import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ZAxis } from "recharts";
import { getOptimization, listOptimizationPoints } from "../api/endpoints";
import type { ParetoOptimization, ParetoPoint } from "../types";

export default function OptimizationDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [opt, setOpt] = useState<ParetoOptimization | null>(null);
  const [points, setPoints] = useState<ParetoPoint[]>([]);
  const [loading, setLoading] = useState(true);

  const load = () => {
    if (!id) return;
    getOptimization(id).then(setOpt);
    listOptimizationPoints(id, 1, 500).then(d => setPoints(d.items)).catch(() => {});
    setLoading(false);
  };

  useEffect(() => { load(); }, [id]);

  if (loading || !opt) return <p>Loading...</p>;

  const pareto = points.filter(p => !p.is_dominated);
  const dominated = points.filter(p => p.is_dominated);
  const chartData = [
    ...pareto.map(p => ({ ...p, cat: "Pareto-optimal" })),
    ...dominated.map(p => ({ ...p, cat: "Dominated" })),
  ];

  return (
    <div>
      <button onClick={() => navigate("/optimizations")} style={backBtn}>← Back</button>
      <h1>🎯 {opt.name || "Optimization"}</h1>

      <div style={{ display: "flex", gap: "1rem", marginBottom: "1.5rem" }}>
        <MetricCard label="Status" value={opt.status} />
        <MetricCard label="Points" value={String(opt.n_points)} />
        <MetricCard label="Pareto-optimal" value={String(pareto.length)} />
        <MetricCard label="Mode" value={opt.mode} />
      </div>

      {opt.status === "completed" && points.length > 0 && (
        <>
          <section style={{ background: "#fff", padding: "1.5rem", borderRadius: 12, marginBottom: "1.5rem" }}>
            <h2>📈 Pareto Frontier</h2>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="total_input" name="Total Input" />
                <YAxis dataKey="total_output" name="Total Output" />
                <ZAxis range={[40, 40]} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Legend />
                <Scatter name="Pareto-optimal" data={chartData.filter(d => d.cat === "Pareto-optimal")} fill="#4361ee" />
                <Scatter name="Dominated" data={chartData.filter(d => d.cat === "Dominated")} fill="#ccc" opacity={0.5} />
              </ScatterChart>
            </ResponsiveContainer>
          </section>

          <section style={{ background: "#fff", padding: "1.5rem", borderRadius: 12, marginBottom: "1.5rem" }}>
            <h2>Efficiency vs Ratio</h2>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="ratio" name="Ratio" />
                <YAxis dataKey="efficiency" name="Efficiency (%)" />
                <ZAxis range={[40, 40]} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Legend />
                <Scatter name="Pareto-optimal" data={chartData.filter(d => d.cat === "Pareto-optimal")} fill="#f72585" />
                <Scatter name="Dominated" data={chartData.filter(d => d.cat === "Dominated")} fill="#ccc" opacity={0.5} />
              </ScatterChart>
            </ResponsiveContainer>
          </section>

          <section style={{ background: "#fff", padding: "1.5rem", borderRadius: 12 }}>
            <h2>Points</h2>
            <div style={{ maxHeight: 400, overflow: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid #eee", textAlign: "left" }}>
                    <th style={thStyle}>Ratio</th>
                    <th style={thStyle}>Input</th>
                    <th style={thStyle}>Output</th>
                    <th style={thStyle}>Efficiency</th>
                    <th style={thStyle}>Pareto?</th>
                  </tr>
                </thead>
                <tbody>
                  {points.map(p => (
                    <tr key={p.id} style={{ borderBottom: "1px solid #eee", opacity: p.is_dominated ? 0.5 : 1 }}>
                      <td style={tdStyle}>{p.ratio.toFixed(4)}</td>
                      <td style={tdStyle}>{p.total_input.toFixed(4)}</td>
                      <td style={tdStyle}>{p.total_output.toFixed(4)}</td>
                      <td style={tdStyle}>{p.efficiency.toFixed(2)}%</td>
                      <td style={tdStyle}>{p.is_dominated ? "❌" : "✅"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      {opt.status === "failed" && <p style={{ color: "red" }}>Failed: {opt.error_message || "Unknown"}</p>}
      {(opt.status === "pending" || opt.status === "processing") && <p>Processing... <button onClick={load} style={btnStyle}>Refresh</button></p>}
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
const btnStyle: React.CSSProperties = { padding: "0.5rem 1rem", borderRadius: 6, border: "none", background: "#4361ee", color: "#fff", cursor: "pointer", marginTop: "0.5rem" };
const thStyle: React.CSSProperties = { padding: "0.5rem", fontSize: "0.85rem", color: "#666" };
const tdStyle: React.CSSProperties = { padding: "0.5rem", fontSize: "0.9rem" };
