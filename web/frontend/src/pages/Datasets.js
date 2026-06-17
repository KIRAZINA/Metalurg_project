import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { listDatasets, uploadDataset, deleteDataset } from "../api/endpoints";
export default function Datasets() {
    const navigate = useNavigate();
    const [datasets, setDatasets] = useState([]);
    const [total, setTotal] = useState(0);
    const [loading, setLoading] = useState(true);
    const [uploading, setUploading] = useState(false);
    const fileRef = useRef(null);
    const nameRef = useRef(null);
    const load = () => {
        setLoading(true);
        listDatasets().then(d => { setDatasets(d.items); setTotal(d.total); }).finally(() => setLoading(false));
    };
    useEffect(() => { load(); }, []);
    const handleUpload = async () => {
        const file = fileRef.current?.files?.[0];
        const name = nameRef.current?.value;
        if (!file || !name)
            return;
        setUploading(true);
        try {
            await uploadDataset(file, name);
            load();
            if (fileRef.current)
                fileRef.current.value = "";
            if (nameRef.current)
                nameRef.current.value = "";
        }
        catch (err) {
            alert(err instanceof Error ? err.message : "Upload failed");
        }
        finally {
            setUploading(false);
        }
    };
    const handleDelete = async (id) => {
        if (!confirm("Delete this dataset?"))
            return;
        try {
            await deleteDataset(id);
            load();
        }
        catch (err) {
            alert(err instanceof Error ? err.message : "Delete failed");
        }
    };
    return (_jsxs("div", { children: [_jsx("h1", { children: "\uD83D\uDCCA Datasets" }), _jsxs("section", { style: { background: "#fff", padding: "1.5rem", borderRadius: 12, marginBottom: "1.5rem" }, children: [_jsx("h2", { children: "Upload New" }), _jsxs("div", { style: { display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }, children: [_jsx("input", { ref: nameRef, type: "text", placeholder: "Dataset name", style: inputStyle }), _jsx("input", { ref: fileRef, type: "file", accept: ".xls,.xlsx", style: inputStyle }), _jsx("button", { onClick: handleUpload, disabled: uploading, style: btnStyle, children: uploading ? "Uploading..." : "Upload" })] })] }), _jsxs("section", { style: { background: "#fff", padding: "1.5rem", borderRadius: 12 }, children: [_jsxs("h2", { children: ["Your Datasets (", total, ")"] }), loading ? (_jsx("p", { children: "Loading..." })) : datasets.length === 0 ? (_jsx("p", { children: "No datasets yet." })) : (_jsxs("table", { style: { width: "100%", borderCollapse: "collapse" }, children: [_jsx("thead", { children: _jsxs("tr", { style: { textAlign: "left", borderBottom: "2px solid #eee" }, children: [_jsx("th", { style: thStyle, children: "Name" }), _jsx("th", { style: thStyle, children: "Status" }), _jsx("th", { style: thStyle, children: "Rows" }), _jsx("th", { style: thStyle, children: "Size" }), _jsx("th", { style: thStyle, children: "Actions" })] }) }), _jsx("tbody", { children: datasets.map(ds => (_jsxs("tr", { style: { borderBottom: "1px solid #eee" }, children: [_jsx("td", { style: tdStyle, children: ds.name }), _jsx("td", { style: tdStyle, children: _jsx(StatusBadge, { status: ds.status }) }), _jsx("td", { style: tdStyle, children: ds.row_count ?? "—" }), _jsxs("td", { style: tdStyle, children: [(ds.file_size_bytes / 1024).toFixed(1), " KB"] }), _jsxs("td", { style: { ...tdStyle, display: "flex", gap: "0.5rem" }, children: [_jsx("button", { onClick: () => navigate(`/datasets/${ds.id}`), style: smallBtn, children: "Open" }), _jsx("button", { onClick: () => handleDelete(ds.id), style: { ...smallBtn, background: "#e74c3c" }, children: "Delete" })] })] }, ds.id))) })] }))] })] }));
}
function StatusBadge({ status }) {
    const colors = { uploading: "#f39c12", processing: "#3498db", ready: "#2ecc71", failed: "#e74c3c" };
    return _jsx("span", { style: { background: colors[status] || "#95a5a6", color: "#fff", padding: "2px 8px", borderRadius: 4, fontSize: "0.8rem" }, children: status });
}
const inputStyle = { padding: "0.5rem", borderRadius: 6, border: "1px solid #ccc", fontSize: "0.9rem" };
const btnStyle = { padding: "0.5rem 1rem", borderRadius: 6, border: "none", background: "#4361ee", color: "#fff", cursor: "pointer", fontSize: "0.9rem" };
const smallBtn = { ...btnStyle, padding: "0.3rem 0.75rem", fontSize: "0.8rem" };
const thStyle = { padding: "0.75rem 0.5rem", fontSize: "0.85rem", color: "#666" };
const tdStyle = { padding: "0.75rem 0.5rem", fontSize: "0.9rem" };
