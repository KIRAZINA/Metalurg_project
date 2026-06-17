import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ZAxis } from "recharts";
import { getOptimization, listOptimizationPoints } from "../api/endpoints";
export default function OptimizationDetail() {
    const { id } = useParams();
    const navigate = useNavigate();
    const [opt, setOpt] = useState(null);
    const [points, setPoints] = useState([]);
    const [loading, setLoading] = useState(true);
    const load = () => {
        if (!id)
            return;
        getOptimization(id).then(setOpt);
        listOptimizationPoints(id, 1, 500).then(d => setPoints(d.items)).catch(() => { });
        setLoading(false);
    };
    useEffect(() => { load(); }, [id]);
    if (loading || !opt)
        return _jsx("p", { children: "Loading..." });
    const pareto = points.filter(p => !p.is_dominated);
    const dominated = points.filter(p => p.is_dominated);
    const chartData = [
        ...pareto.map(p => ({ ...p, cat: "Pareto-optimal" })),
        ...dominated.map(p => ({ ...p, cat: "Dominated" })),
    ];
    return (_jsxs("div", { children: [_jsx("button", { onClick: () => navigate("/optimizations"), style: backBtn, children: "\u2190 Back" }), _jsxs("h1", { children: ["\uD83C\uDFAF ", opt.name || "Optimization"] }), _jsxs("div", { style: { display: "flex", gap: "1rem", marginBottom: "1.5rem" }, children: [_jsx(MetricCard, { label: "Status", value: opt.status }), _jsx(MetricCard, { label: "Points", value: String(opt.n_points) }), _jsx(MetricCard, { label: "Pareto-optimal", value: String(pareto.length) }), _jsx(MetricCard, { label: "Mode", value: opt.mode })] }), opt.status === "completed" && points.length > 0 && (_jsxs(_Fragment, { children: [_jsxs("section", { style: { background: "#fff", padding: "1.5rem", borderRadius: 12, marginBottom: "1.5rem" }, children: [_jsx("h2", { children: "\uD83D\uDCC8 Pareto Frontier" }), _jsx(ResponsiveContainer, { width: "100%", height: 400, children: _jsxs(ScatterChart, { margin: { top: 20, right: 20, bottom: 20, left: 20 }, children: [_jsx(CartesianGrid, { strokeDasharray: "3 3" }), _jsx(XAxis, { dataKey: "total_input", name: "Total Input" }), _jsx(YAxis, { dataKey: "total_output", name: "Total Output" }), _jsx(ZAxis, { range: [40, 40] }), _jsx(Tooltip, { cursor: { strokeDasharray: "3 3" } }), _jsx(Legend, {}), _jsx(Scatter, { name: "Pareto-optimal", data: chartData.filter(d => d.cat === "Pareto-optimal"), fill: "#4361ee" }), _jsx(Scatter, { name: "Dominated", data: chartData.filter(d => d.cat === "Dominated"), fill: "#ccc", opacity: 0.5 })] }) })] }), _jsxs("section", { style: { background: "#fff", padding: "1.5rem", borderRadius: 12, marginBottom: "1.5rem" }, children: [_jsx("h2", { children: "Efficiency vs Ratio" }), _jsx(ResponsiveContainer, { width: "100%", height: 300, children: _jsxs(ScatterChart, { margin: { top: 20, right: 20, bottom: 20, left: 20 }, children: [_jsx(CartesianGrid, { strokeDasharray: "3 3" }), _jsx(XAxis, { dataKey: "ratio", name: "Ratio" }), _jsx(YAxis, { dataKey: "efficiency", name: "Efficiency (%)" }), _jsx(ZAxis, { range: [40, 40] }), _jsx(Tooltip, { cursor: { strokeDasharray: "3 3" } }), _jsx(Legend, {}), _jsx(Scatter, { name: "Pareto-optimal", data: chartData.filter(d => d.cat === "Pareto-optimal"), fill: "#f72585" }), _jsx(Scatter, { name: "Dominated", data: chartData.filter(d => d.cat === "Dominated"), fill: "#ccc", opacity: 0.5 })] }) })] }), _jsxs("section", { style: { background: "#fff", padding: "1.5rem", borderRadius: 12 }, children: [_jsx("h2", { children: "Points" }), _jsx("div", { style: { maxHeight: 400, overflow: "auto" }, children: _jsxs("table", { style: { width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }, children: [_jsx("thead", { children: _jsxs("tr", { style: { borderBottom: "2px solid #eee", textAlign: "left" }, children: [_jsx("th", { style: thStyle, children: "Ratio" }), _jsx("th", { style: thStyle, children: "Input" }), _jsx("th", { style: thStyle, children: "Output" }), _jsx("th", { style: thStyle, children: "Efficiency" }), _jsx("th", { style: thStyle, children: "Pareto?" })] }) }), _jsx("tbody", { children: points.map(p => (_jsxs("tr", { style: { borderBottom: "1px solid #eee", opacity: p.is_dominated ? 0.5 : 1 }, children: [_jsx("td", { style: tdStyle, children: p.ratio.toFixed(4) }), _jsx("td", { style: tdStyle, children: p.total_input.toFixed(4) }), _jsx("td", { style: tdStyle, children: p.total_output.toFixed(4) }), _jsxs("td", { style: tdStyle, children: [p.efficiency.toFixed(2), "%"] }), _jsx("td", { style: tdStyle, children: p.is_dominated ? "❌" : "✅" })] }, p.id))) })] }) })] })] })), opt.status === "failed" && _jsxs("p", { style: { color: "red" }, children: ["Failed: ", opt.error_message || "Unknown"] }), (opt.status === "pending" || opt.status === "processing") && _jsxs("p", { children: ["Processing... ", _jsx("button", { onClick: load, style: btnStyle, children: "Refresh" })] })] }));
}
function MetricCard({ label, value }) {
    return (_jsxs("div", { style: { flex: 1, padding: "1rem", background: "#fff", borderRadius: 8, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }, children: [_jsx("p", { style: { margin: 0, fontSize: "0.8rem", color: "#666" }, children: label }), _jsx("p", { style: { margin: "0.25rem 0 0", fontSize: "1.25rem", fontWeight: 600 }, children: value })] }));
}
const backBtn = { background: "none", border: "1px solid #ccc", borderRadius: 6, padding: "0.4rem 0.75rem", cursor: "pointer", marginBottom: "1rem" };
const btnStyle = { padding: "0.5rem 1rem", borderRadius: 6, border: "none", background: "#4361ee", color: "#fff", cursor: "pointer", marginTop: "0.5rem" };
const thStyle = { padding: "0.5rem", fontSize: "0.85rem", color: "#666" };
const tdStyle = { padding: "0.5rem", fontSize: "0.9rem" };
