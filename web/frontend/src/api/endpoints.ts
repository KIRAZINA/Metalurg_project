import { api } from "./client";
import type {
  AsyncTask,
  AuthTokens,
  Dataset,
  DatasetList,
  PaginatedResponse,
  ParetoOptimization,
  ParetoPoint,
  RegressionModel,
  User,
} from "../types";

export function login(email: string, password: string) {
  return api.post<AuthTokens & { user: User }>("/auth/login", { email, password });
}

export function register(email: string, password: string, fullName?: string) {
  return api.post<User>("/auth/register", { email, password, full_name: fullName });
}

export function getMe() {
  return api.get<User>("/auth/me");
}

export function refreshToken(refresh: string) {
  return api.post<AuthTokens>("/auth/refresh", { refresh_token: refresh });
}

export function listDatasets(page = 1, pageSize = 20) {
  return api.get<DatasetList>("/datasets", { page, page_size: pageSize });
}

export function getDataset(id: string) {
  return api.get<Dataset>(`/datasets/${id}`);
}

export function uploadDataset(file: File, name: string, description?: string) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("name", name);
  if (description) fd.append("description", description);
  return api.post<Dataset>("/datasets", fd, undefined, true);
}

export function updateDataset(id: string, data: { name?: string; description?: string }) {
  return api.patch<Dataset>(`/datasets/${id}`, data);
}

export function deleteDataset(id: string) {
  return api.delete<void>(`/datasets/${id}`);
}

export function triggerRegression(datasetId: string) {
  return api.post<{ task_id: string; status: string }>(`/datasets/${datasetId}/regressions`);
}

export function getDatasetRegressions(datasetId: string) {
  return api.get<{ items: RegressionModel[] }>(`datasets/${datasetId}/regressions`);
}

export function getRegression(id: string) {
  return api.get<RegressionModel>(`/regressions/${id}`);
}

export function listOptimizations(page = 1, pageSize = 20) {
  return api.get<PaginatedResponse<ParetoOptimization>>("/optimizations", { page, page_size: pageSize });
}

export function createOptimization(params: {
  dataset_id: string;
  name?: string;
  targets: string;
  mode?: string;
  n_points?: number;
}) {
  return api.post<ParetoOptimization>("/optimizations", undefined, {
    dataset_id: params.dataset_id,
    name: params.name,
    targets: params.targets,
    mode: params.mode,
    n_points: params.n_points,
  });
}

export function getOptimization(id: string) {
  return api.get<ParetoOptimization>(`/optimizations/${id}`);
}

export function listOptimizationPoints(id: string, page = 1, pageSize = 100) {
  return api.get<PaginatedResponse<ParetoPoint>>(`/optimizations/${id}/points`, { page, page_size: pageSize });
}

export function deleteOptimization(id: string) {
  return api.delete<void>(`/optimizations/${id}`);
}

export function listTasks(page = 1, pageSize = 20) {
  return api.get<PaginatedResponse<AsyncTask>>("/tasks", { page, page_size: pageSize });
}

export function getTask(id: string) {
  return api.get<AsyncTask>(`/tasks/${id}`);
}
