import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
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
    return (_jsxs("div", { children: [_jsx("h1", { children: "Dashboard" }), _jsxs("p", { children: ["Welcome, ", user?.full_name || user?.email, "!"] }), loading ? (_jsx("p", { children: "Loading stats..." })) : (_jsxs("div", { style: { display: "flex", gap: "1rem", marginTop: "1.5rem" }, children: [_jsx(StatCard, { label: "Datasets", value: stats.datasets, color: "#4361ee" }), _jsx(StatCard, { label: "Optimizations", value: stats.optimizations, color: "#f72585" }), _jsx(StatCard, { label: "Tasks", value: stats.tasks, color: "#4cc9f0" })] })), _jsxs("div", { style: { marginTop: "2rem", lineHeight: 2 }, children: [_jsx("h2", { children: "Quick Actions" }), _jsxs("ul", { children: [_jsx("li", { children: _jsx("a", { href: "/datasets", children: "\uD83D\uDCCA Browse Datasets" }) }), _jsx("li", { children: _jsx("a", { href: "/optimizations", children: "\uD83C\uDFAF Create Optimization" }) }), _jsx("li", { children: _jsx("a", { href: "/tasks", children: "\uD83D\uDCCB View Tasks" }) })] })] })] }));
}
function StatCard({ label, value, color }) {
    return (_jsxs("div", { style: { flex: 1, padding: "1.5rem", background: "#fff", borderRadius: 12, boxShadow: "0 2px 8px rgba(0,0,0,0.06)", borderTop: `4px solid ${color}` }, children: [_jsx("h3", { style: { margin: 0, fontSize: "0.9rem", color: "#666" }, children: label }), _jsx("p", { style: { margin: "0.5rem 0 0", fontSize: "2rem", fontWeight: 700 }, children: value })] }));
}
