import { api } from "./client";
export function login(email, password) {
    return api.post("/auth/login", { email, password });
}
export function register(email, password, fullName) {
    return api.post("/auth/register", { email, password, full_name: fullName });
}
export function getMe() {
    return api.get("/auth/me");
}
export function refreshToken(refresh) {
    return api.post("/auth/refresh", { refresh_token: refresh });
}
export function listDatasets(page = 1, pageSize = 20) {
    return api.get("/datasets", { page, page_size: pageSize });
}
export function getDataset(id) {
    return api.get(`/datasets/${id}`);
}
export function uploadDataset(file, name, description) {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("name", name);
    if (description)
        fd.append("description", description);
    return api.post("/datasets", fd, undefined, true);
}
export function updateDataset(id, data) {
    return api.patch(`/datasets/${id}`, data);
}
export function deleteDataset(id) {
    return api.delete(`/datasets/${id}`);
}
export function triggerRegression(datasetId) {
    return api.post(`/datasets/${datasetId}/regressions`);
}
export function getDatasetRegressions(datasetId) {
    return api.get(`datasets/${datasetId}/regressions`);
}
export function getRegression(id) {
    return api.get(`/regressions/${id}`);
}
export function listOptimizations(page = 1, pageSize = 20) {
    return api.get("/optimizations", { page, page_size: pageSize });
}
export function createOptimization(params) {
    return api.post("/optimizations", undefined, {
        dataset_id: params.dataset_id,
        name: params.name,
        targets: params.targets,
        mode: params.mode,
        n_points: params.n_points,
    });
}
export function getOptimization(id) {
    return api.get(`/optimizations/${id}`);
}
export function listOptimizationPoints(id, page = 1, pageSize = 100) {
    return api.get(`/optimizations/${id}/points`, { page, page_size: pageSize });
}
export function deleteOptimization(id) {
    return api.delete(`/optimizations/${id}`);
}
export function listTasks(page = 1, pageSize = 20) {
    return api.get("/tasks", { page, page_size: pageSize });
}
export function getTask(id) {
    return api.get(`/tasks/${id}`);
}
