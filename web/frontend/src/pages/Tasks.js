import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { listTasks } from "../api/endpoints";
export default function Tasks() {
    const [tasks, setTasks] = useState([]);
    const [loading, setLoading] = useState(true);
    const load = () => {
        setLoading(true);
        listTasks().then(d => setTasks(d.items)).finally(() => setLoading(false));
    };
    useEffect(() => { load(); }, []);
    const statusIcon = {
        PENDING: "🔵",
        PROGRESS: "🟡",
        SUCCESS: "🟢",
        FAILURE: "🔴",
    };
    return (_jsxs("div", { children: [_jsxs("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center" }, children: [_jsx("h1", { children: "\uD83D\uDCCB Tasks" }), _jsx("button", { onClick: load, style: btnStyle, children: "Refresh" })] }), loading ? (_jsx("p", { children: "Loading..." })) : tasks.length === 0 ? (_jsx("p", { children: "No tasks yet." })) : (_jsx("div", { style: { display: "flex", flexDirection: "column", gap: "0.75rem" }, children: tasks.map(t => (_jsxs("div", { style: { background: "#fff", padding: "1rem", borderRadius: 8, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }, children: [_jsxs("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center" }, children: [_jsxs("div", { children: [_jsxs("strong", { children: [statusIcon[t.status] || "⚪", " ", t.task_type] }), _jsx("span", { style: { marginLeft: "1rem", fontSize: "0.85rem", color: "#666" }, children: t.status })] }), _jsx("span", { style: { fontSize: "0.8rem", color: "#999" }, children: new Date(t.created_at).toLocaleString() })] }), _jsx("div", { style: { marginTop: "0.5rem", background: "#eee", borderRadius: 4, height: 8, overflow: "hidden" }, children: _jsx("div", { style: { width: `${t.progress}%`, background: t.status === "FAILURE" ? "#e74c3c" : "#4361ee", height: "100%", borderRadius: 4, transition: "width 0.3s" } }) }), _jsxs("div", { style: { display: "flex", justifyContent: "space-between", marginTop: "0.25rem" }, children: [_jsxs("span", { style: { fontSize: "0.8rem", color: "#666" }, children: [t.progress, "%"] }), _jsxs("span", { style: { fontSize: "0.75rem", color: "#999", fontFamily: "monospace" }, children: ["ID: ", t.id.slice(0, 8), "\u2026"] })] }), t.error_message && _jsx("p", { style: { color: "red", fontSize: "0.85rem", margin: "0.5rem 0 0" }, children: t.error_message })] }, t.id))) }))] }));
}
const btnStyle = { padding: "0.5rem 1rem", borderRadius: 6, border: "none", background: "#4361ee", color: "#fff", cursor: "pointer" };
