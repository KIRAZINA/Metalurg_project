import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getDataset, updateDataset, getDatasetRegressions, triggerRegression } from "../api/endpoints";
export default function DatasetDetail() {
    const { id } = useParams();
    const navigate = useNavigate();
    const [ds, setDs] = useState(null);
    const [regressions, setRegressions] = useState([]);
    const [name, setName] = useState("");
    const [description, setDescription] = useState("");
    const [loading, setLoading] = useState(true);
    const [running, setRunning] = useState(false);
    useEffect(() => {
        if (!id)
            return;
        getDataset(id).then(d => { setDs(d); setName(d.name); setDescription(d.description || ""); });
        getDatasetRegressions(id).then(r => setRegressions(r.items)).catch(() => { });
        setLoading(false);
    }, [id]);
    const handleUpdate = async () => {
        if (!id)
            return;
        try {
            const d = await updateDataset(id, { name, description: description || undefined });
            setDs(d);
        }
        catch (err) {
            alert(err);
        }
    };
    const handleRunRegression = async () => {
        if (!id)
            return;
        setRunning(true);
        try {
            const result = await triggerRegression(id);
            alert(`Task started: ${result.task_id}`);
        }
        catch (err) {
            alert(err);
        }
        finally {
            setRunning(false);
        }
    };
    if (loading || !ds)
        return _jsx("p", { children: "Loading..." });
    return (_jsxs("div", { children: [_jsx("button", { onClick: () => navigate("/datasets"), style: backBtn, children: "\u2190 Back" }), _jsxs("h1", { children: ["\uD83D\uDCCA ", ds.name] }), _jsxs("div", { style: { display: "flex", gap: "1rem", marginBottom: "1.5rem" }, children: [_jsx(MetricCard, { label: "Status", value: ds.status }), _jsx(MetricCard, { label: "Rows", value: ds.row_count?.toString() || "—" }), _jsx(MetricCard, { label: "Columns", value: ds.column_count?.toString() || "—" }), _jsx(MetricCard, { label: "Size", value: `${(ds.file_size_bytes / 1024).toFixed(1)} KB` })] }), _jsxs("section", { style: { background: "#fff", padding: "1.5rem", borderRadius: 12, marginBottom: "1.5rem" }, children: [_jsx("h2", { children: "Details" }), _jsxs("p", { children: [_jsx("strong", { children: "File:" }), " ", ds.original_filename] }), ds.error_message && _jsx("p", { style: { color: "red" }, children: ds.error_message }), _jsxs("div", { style: { display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }, children: [_jsx("input", { value: name, onChange: e => setName(e.target.value), style: inputStyle }), _jsx("input", { value: description, onChange: e => setDescription(e.target.value), placeholder: "Description", style: inputStyle }), _jsx("button", { onClick: handleUpdate, style: btnStyle, children: "Update" })] })] }), _jsxs("section", { style: { background: "#fff", padding: "1.5rem", borderRadius: 12, marginBottom: "1.5rem" }, children: [_jsx("h2", { children: "\uD83D\uDD2C Regressions" }), _jsx("button", { onClick: handleRunRegression, disabled: running, style: btnStyle, children: running ? "Running..." : "Run Regression" }), regressions.length === 0 ? (_jsx("p", { style: { marginTop: "1rem" }, children: "No regressions yet." })) : (_jsxs("table", { style: { width: "100%", borderCollapse: "collapse", marginTop: "1rem" }, children: [_jsx("thead", { children: _jsxs("tr", { style: { borderBottom: "2px solid #eee", textAlign: "left" }, children: [_jsx("th", { style: thStyle, children: "X \u2192 Y" }), _jsx("th", { style: thStyle, children: "R\u00B2" }), _jsx("th", { style: thStyle, children: "Slope" }), _jsx("th", { style: thStyle, children: "Intercept" }), _jsx("th", { style: thStyle, children: "Confidence" })] }) }), _jsx("tbody", { children: regressions.map(r => (_jsxs("tr", { style: { borderBottom: "1px solid #eee" }, children: [_jsxs("td", { style: tdStyle, children: [r.x_column, " \u2192 ", r.y_column] }), _jsx("td", { style: tdStyle, children: r.r_squared.toFixed(4) }), _jsx("td", { style: tdStyle, children: r.slope.toFixed(6) }), _jsx("td", { style: tdStyle, children: r.intercept.toFixed(6) }), _jsx("td", { style: tdStyle, children: r.confidence })] }, r.id))) })] }))] })] }));
}
function MetricCard({ label, value }) {
    return (_jsxs("div", { style: { flex: 1, padding: "1rem", background: "#fff", borderRadius: 8, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }, children: [_jsx("p", { style: { margin: 0, fontSize: "0.8rem", color: "#666" }, children: label }), _jsx("p", { style: { margin: "0.25rem 0 0", fontSize: "1.25rem", fontWeight: 600 }, children: value })] }));
}
const backBtn = { background: "none", border: "1px solid #ccc", borderRadius: 6, padding: "0.4rem 0.75rem", cursor: "pointer", marginBottom: "1rem" };
const inputStyle = { padding: "0.5rem", borderRadius: 6, border: "1px solid #ccc", fontSize: "0.9rem", flex: 1, minWidth: 200 };
const btnStyle = { padding: "0.5rem 1rem", borderRadius: 6, border: "none", background: "#4361ee", color: "#fff", cursor: "pointer" };
const thStyle = { padding: "0.5rem", fontSize: "0.85rem", color: "#666" };
const tdStyle = { padding: "0.5rem", fontSize: "0.9rem" };
