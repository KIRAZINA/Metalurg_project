const API_BASE = "/api/v1";
let accessToken = null;
let refreshToken = null;
let onLogout = null;
export function setTokens(access, refresh) {
    accessToken = access;
    refreshToken = refresh;
}
export function clearTokens() {
    accessToken = null;
    refreshToken = null;
}
export function setLogoutHandler(fn) {
    onLogout = fn;
}
export class ApiError extends Error {
    status;
    constructor(status, message) {
        super(message);
        this.status = status;
        this.name = "ApiError";
    }
}
export function getAccessToken() {
    return accessToken;
}
async function request(path, opts = {}) {
    const { method = "GET", body, params, isFormData } = opts;
    const url = new URL(`${API_BASE}${path}`, window.location.origin);
    if (params) {
        for (const [k, v] of Object.entries(params)) {
            if (v !== undefined)
                url.searchParams.set(k, String(v));
        }
    }
    const exec = () => {
        const headers = {};
        if (accessToken)
            headers["Authorization"] = `Bearer ${accessToken}`;
        if (!isFormData)
            headers["Content-Type"] = "application/json";
        return fetch(url.toString(), {
            method,
            headers,
            body: isFormData ? body : body ? JSON.stringify(body) : undefined,
        });
    };
    let res = await exec();
    if (res.status === 401 && refreshToken) {
        const refreshRes = await fetch(`${API_BASE}/auth/refresh`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ refresh_token: refreshToken }),
        });
        if (refreshRes.ok) {
            const data = await refreshRes.json();
            accessToken = data.access_token;
            refreshToken = data.refresh_token ?? refreshToken;
            res = await exec();
        }
        else {
            clearTokens();
            onLogout?.();
            throw new ApiError(401, "Session expired");
        }
    }
    if (!res.ok) {
        const text = await res.text();
        throw new ApiError(res.status, text);
    }
    if (res.status === 204)
        return undefined;
    return res.json();
}
export const api = {
    get: (path, params) => request(path, { params }),
    post: (path, body, params, isFormData) => request(path, { method: "POST", body, params, isFormData }),
    patch: (path, body) => request(path, { method: "PATCH", body }),
    delete: (path) => request(path, { method: "DELETE" }),
};
