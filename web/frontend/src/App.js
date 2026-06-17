import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import { Layout } from "./components/Layout";
import { ProtectedRoute } from "./components/ProtectedRoute";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import Datasets from "./pages/Datasets";
import DatasetDetail from "./pages/DatasetDetail";
import Optimizations from "./pages/Optimizations";
import OptimizationDetail from "./pages/OptimizationDetail";
import Tasks from "./pages/Tasks";
export default function App() {
    return (_jsx(BrowserRouter, { children: _jsx(AuthProvider, { children: _jsxs(Routes, { children: [_jsx(Route, { path: "/login", element: _jsx(Login, {}) }), _jsx(Route, { path: "/register", element: _jsx(Register, {}) }), _jsxs(Route, { element: _jsx(ProtectedRoute, { children: _jsx(Layout, {}) }), children: [_jsx(Route, { path: "/", element: _jsx(Dashboard, {}) }), _jsx(Route, { path: "/datasets", element: _jsx(Datasets, {}) }), _jsx(Route, { path: "/datasets/:id", element: _jsx(DatasetDetail, {}) }), _jsx(Route, { path: "/optimizations", element: _jsx(Optimizations, {}) }), _jsx(Route, { path: "/optimizations/:id", element: _jsx(OptimizationDetail, {}) }), _jsx(Route, { path: "/tasks", element: _jsx(Tasks, {}) })] })] }) }) }));
}
