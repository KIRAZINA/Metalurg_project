import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listOptimizations, listDatasets, createOptimization, deleteOptimization } from "../api/endpoints";
export default function Optimizations() {
    const navigate = useNavigate();
    const [items, setItems] = useState([]);
    const [total, setTotal] = useState(0);
    const [loading, setLoading] = useState(true);
    const [creating, setCreating] = useState(false);
    const [dsList, setDsList] = useState([]);
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
        if (!selectedDs)
            return alert("Select a dataset");
        setCreating(true);
        try {
            const result = await createOptimization({ dataset_id: selectedDs, name: optName || undefined, targets: targetsStr, n_points: nPoints });
            navigate(`/optimizations/${result.id}`);
        }
        catch (err) {
            alert(err);
        }
        finally {
            setCreating(false);
        }
    };
    const handleDelete = async (id) => {
        if (!confirm("Delete this optimization?"))
            return;
        try {
            await deleteOptimization(id);
            load();
        }
        catch (err) {
            alert(err);
        }
    };
    return (_jsxs("div", { children: [_jsx("h1", { children: "\uD83C\uDFAF Optimizations" }), _jsxs("section", { style: { background: "#fff", padding: "1.5rem", borderRadius: 12, marginBottom: "1.5rem" }, children: [_jsx("h2", { children: "Create New" }), _jsxs("div", { style: { display: "flex", flexDirection: "column", gap: "0.75rem" }, children: [_jsxs("select", { value: selectedDs, onChange: e => setSelectedDs(e.target.value), style: inputStyle, children: [_jsx("option", { value: "", children: "\u2014 Select dataset \u2014" }), dsList.map(ds => _jsx("option", { value: ds.id, children: ds.name }, ds.id))] }), _jsx("input", { value: optName, onChange: e => setOptName(e.target.value), placeholder: "Optimization name (optional)", style: inputStyle }), _jsxs("label", { children: ["Number of points: ", nPoints] }), _jsx("input", { type: "range", min: 10, max: 500, value: nPoints, onChange: e => setNPoints(Number(e.target.value)) }), _jsx("textarea", { value: targetsStr, onChange: e => setTargetsStr(e.target.value), rows: 6, style: { ...inputStyle, fontFamily: "monospace" } }), _jsx("button", { onClick: handleCreate, disabled: creating, style: btnStyle, children: creating ? "Creating..." : "Create & Run" })] })] }), _jsxs("section", { style: { background: "#fff", padding: "1.5rem", borderRadius: 12 }, children: [_jsxs("h2", { children: ["Your Optimizations (", total, ")"] }), loading ? _jsx("p", { children: "Loading..." }) : items.length === 0 ? _jsx("p", { children: "None yet." }) : (_jsxs("table", { style: { width: "100%", borderCollapse: "collapse" }, children: [_jsx("thead", { children: _jsxs("tr", { style: { borderBottom: "2px solid #eee", textAlign: "left" }, children: [_jsx("th", { style: thStyle, children: "Name" }), _jsx("th", { style: thStyle, children: "Status" }), _jsx("th", { style: thStyle, children: "Points" }), _jsx("th", { style: thStyle, children: "Actions" })] }) }), _jsx("tbody", { children: items.map(o => (_jsxs("tr", { style: { borderBottom: "1px solid #eee" }, children: [_jsx("td", { style: tdStyle, children: o.name || "Unnamed" }), _jsx("td", { style: tdStyle, children: o.status }), _jsx("td", { style: tdStyle, children: o.n_points }), _jsxs("td", { style: tdStyle, children: [_jsx("button", { onClick: () => navigate(`/optimizations/${o.id}`), style: smallBtn, children: "Open" }), _jsx("button", { onClick: () => handleDelete(o.id), style: { ...smallBtn, background: "#e74c3c", marginLeft: "0.5rem" }, children: "Delete" })] })] }, o.id))) })] }))] })] }));
}
const inputStyle = { padding: "0.5rem", borderRadius: 6, border: "1px solid #ccc", fontSize: "0.9rem" };
const btnStyle = { padding: "0.5rem 1rem", borderRadius: 6, border: "none", background: "#4361ee", color: "#fff", cursor: "pointer" };
const smallBtn = { ...btnStyle, padding: "0.3rem 0.75rem", fontSize: "0.8rem" };
const thStyle = { padding: "0.5rem", fontSize: "0.85rem", color: "#666" };
const tdStyle = { padding: "0.5rem", fontSize: "0.9rem" };
