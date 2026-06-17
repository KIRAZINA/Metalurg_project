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
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/datasets/:id" element={<DatasetDetail />} />
            <Route path="/optimizations" element={<Optimizations />} />
            <Route path="/optimizations/:id" element={<OptimizationDetail />} />
            <Route path="/tasks" element={<Tasks />} />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
