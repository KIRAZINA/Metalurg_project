from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st

from config import API_URL


@dataclass
class AuthState:
    access_token: str
    refresh_token: str


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    auth = st.session_state.get("auth")
    if auth:
        headers["Authorization"] = f"Bearer {auth.access_token}"
    return headers


def _handle_response(response: requests.Response) -> Any:
    if response.status_code == 401:
        auth = st.session_state.get("auth")
        if auth:
            refresh_resp = requests.post(
                f"{API_URL}/auth/refresh",
                json={"refresh_token": auth.refresh_token},
            )
            if refresh_resp.status_code == 200:
                data = refresh_resp.json()
                st.session_state.auth = AuthState(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token", auth.refresh_token),
                )
                response = requests.request(
                    method=response.request.method or "GET",
                    url=response.request.url or "",
                    headers=_headers(),
                    data=response.request.body,
                )
            else:
                del st.session_state.auth
                st.rerun()
    response.raise_for_status()
    if response.status_code == 204:
        return None
    return response.json()


def register(email: str, password: str, full_name: str | None = None) -> dict:
    body = {"email": email, "password": password}
    if full_name:
        body["full_name"] = full_name
    resp = requests.post(f"{API_URL}/auth/register", json=body)
    return _handle_response(resp)


def login(email: str, password: str) -> dict:
    resp = requests.post(
        f"{API_URL}/auth/login",
        json={"email": email, "password": password},
    )
    return _handle_response(resp)


def get_me() -> dict:
    resp = requests.get(f"{API_URL}/auth/me", headers=_headers())
    return _handle_response(resp)


def list_datasets(page: int = 1, page_size: int = 20) -> dict:
    resp = requests.get(
        f"{API_URL}/datasets",
        params={"page": page, "page_size": page_size},
        headers=_headers(),
    )
    return _handle_response(resp)


def get_dataset(dataset_id: str) -> dict:
    resp = requests.get(f"{API_URL}/datasets/{dataset_id}", headers=_headers())
    return _handle_response(resp)


def upload_dataset(file, name: str, description: str | None = None) -> dict:
    files = {"file": (file.name, file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    data = {"name": name}
    if description:
        data["description"] = description
    resp = requests.post(f"{API_URL}/datasets", files=files, data=data, headers={"Authorization": _headers()["Authorization"]})
    resp.raise_for_status()
    return resp.json()


def delete_dataset(dataset_id: str) -> None:
    resp = requests.delete(f"{API_URL}/datasets/{dataset_id}", headers=_headers())
    _handle_response(resp)


def update_dataset(dataset_id: str, name: str | None = None, description: str | None = None) -> dict:
    body: dict[str, str] = {}
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    resp = requests.patch(f"{API_URL}/datasets/{dataset_id}", json=body, headers=_headers())
    return _handle_response(resp)


def trigger_regression(dataset_id: str) -> dict:
    resp = requests.post(f"{API_URL}/datasets/{dataset_id}/regressions", headers=_headers())
    return _handle_response(resp)


def list_regressions(dataset_id: str) -> dict:
    resp = requests.get(f"{API_URL}/datasets/{dataset_id}/regressions", headers=_headers())
    return _handle_response(resp)


def get_regression(regression_id: str) -> dict:
    resp = requests.get(f"{API_URL}/regressions/{regression_id}", headers=_headers())
    return _handle_response(resp)


def list_optimizations(page: int = 1, page_size: int = 20) -> dict:
    resp = requests.get(
        f"{API_URL}/optimizations",
        params={"page": page, "page_size": page_size},
        headers=_headers(),
    )
    return _handle_response(resp)


def create_optimization(
    dataset_id: str,
    targets: dict[str, Any],
    name: str | None = None,
    mode: str = "after",
    n_points: int = 100,
) -> dict:
    params = {
        "dataset_id": dataset_id,
        "targets": json.dumps(targets),
        "mode": mode,
        "n_points": n_points,
    }
    if name:
        params["name"] = name
    resp = requests.post(f"{API_URL}/optimizations", params=params, headers=_headers())
    return _handle_response(resp)


def get_optimization(optimization_id: str) -> dict:
    resp = requests.get(f"{API_URL}/optimizations/{optimization_id}", headers=_headers())
    return _handle_response(resp)


def list_optimization_points(optimization_id: str, page: int = 1, page_size: int = 100) -> dict:
    resp = requests.get(
        f"{API_URL}/optimizations/{optimization_id}/points",
        params={"page": page, "page_size": page_size},
        headers=_headers(),
    )
    return _handle_response(resp)


def delete_optimization(optimization_id: str) -> None:
    resp = requests.delete(f"{API_URL}/optimizations/{optimization_id}", headers=_headers())
    _handle_response(resp)


def list_tasks(page: int = 1, page_size: int = 20) -> dict:
    resp = requests.get(
        f"{API_URL}/tasks",
        params={"page": page, "page_size": page_size},
        headers=_headers(),
    )
    return _handle_response(resp)


def get_task(task_id: str) -> dict:
    resp = requests.get(f"{API_URL}/tasks/{task_id}", headers=_headers())
    return _handle_response(resp)
