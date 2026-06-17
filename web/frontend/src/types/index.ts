export interface User {
  id: string;
  email: string;
  full_name: string | null;
  role: string;
  is_active: boolean;
  created_at: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface Dataset {
  id: string;
  name: string;
  description: string | null;
  original_filename: string;
  file_size_bytes: number;
  row_count: number | null;
  column_count: number | null;
  status: string;
  error_message: string | null;
  created_at: string;
}

export interface DatasetList {
  items: Dataset[];
  total: number;
  page: number;
  page_size: number;
}

export interface RegressionModel {
  id: string;
  dataset_id: string;
  x_column: string;
  y_column: string;
  slope: number;
  intercept: number;
  r_squared: number;
  p_value: number;
  std_err: number;
  ci_lower: number;
  ci_upper: number;
  n_obs: number;
  x_min: number;
  x_max: number;
  confidence: string;
  is_feasible: boolean | null;
  created_at: string;
}

export interface ParetoOptimization {
  id: string;
  dataset_id: string;
  user_id: string;
  name: string | null;
  targets: Record<string, { x_column: string; target_value: number }>;
  mode: string;
  n_points: number;
  status: string;
  error_message: string | null;
  created_at: string;
}

export interface ParetoPoint {
  id: string;
  ratio: number;
  total_input: number;
  total_output: number;
  efficiency: number;
  inputs: Record<string, number>;
  outputs: Record<string, number>;
  is_dominated: boolean;
  rank: number | null;
}

export interface AsyncTask {
  id: string;
  user_id: string;
  task_type: string;
  status: string;
  progress: number;
  result_ref: Record<string, unknown> | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}
