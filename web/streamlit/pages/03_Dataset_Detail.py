import streamlit as st

from api_client import (
    get_dataset,
    get_regression,
    list_regressions,
    trigger_regression,
    update_dataset,
)


def dataset_detail():
    if "auth" not in st.session_state:
        st.warning("Please log in first")
        st.page_link("pages/01_Login.py", label="Go to Login")
        return

    dataset_id = st.session_state.get("selected_dataset_id")
    if not dataset_id:
        st.info("No dataset selected")
        if st.button("← Back to Datasets"):
            st.switch_page("pages/02_Datasets.py")
        return

    try:
        ds = get_dataset(dataset_id)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        if st.button("← Back"):
            st.switch_page("pages/02_Datasets.py")
        return

    st.title(f"📊 {ds['name']}")
    st.caption(f"ID: `{ds['id']}`")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", ds["status"])
    with col2:
        st.metric("Rows", ds.get("row_count") or "—")
    with col3:
        st.metric("Columns", ds.get("column_count") or "—")

    with st.expander("Details", expanded=True):
        st.write(f"**File:** {ds.get('original_filename', 'N/A')}")
        st.write(f"**Size:** {ds.get('file_size_bytes', 0):,} bytes")
        if ds.get("description"):
            st.write(f"**Description:** {ds['description']}")
        if ds.get("error_message"):
            st.error(ds["error_message"])
        with st.form("edit_dataset"):
            new_name = st.text_input("Name", value=ds["name"])
            new_desc = st.text_area("Description", value=ds.get("description") or "")
            if st.form_submit_button("💾 Update", use_container_width=True):
                try:
                    result = update_dataset(dataset_id, name=new_name, description=new_desc or None)
                    st.success("Updated")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.divider()
    st.subheader("🔬 Regression Analysis")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("▶️ Run Regression", type="primary", use_container_width=True):
            try:
                task_data = trigger_regression(dataset_id)
                st.success(f"Task started: `{task_data['task_id']}`")
                st.session_state.last_task_id = task_data["task_id"]
            except Exception as e:
                st.error(f"Failed to trigger regression: {e}")

    with col_b:
        if st.button("🔄 Refresh Regressions", use_container_width=True):
            st.rerun()

    try:
        reg_data = list_regressions(dataset_id)
        models = reg_data.get("items", [])
        if not models:
            st.info("No regression results yet. Click 'Run Regression' to start.")
        else:
            for model in models:
                with st.expander(f"{model['x_column']} → {model['y_column']}  (R²={model['r_squared']:.4f})"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Slope", f"{model['slope']:.6f}")
                        st.metric("Intercept", f"{model['intercept']:.6f}")
                        st.metric("R²", f"{model['r_squared']:.4f}")
                        st.metric("P-value", f"{model['p_value']:.6e}")
                    with c2:
                        st.metric("Std Err", f"{model['std_err']:.6f}")
                        st.metric("Confidence", model["confidence"])
                        st.metric("Feasible", "✅" if model.get("is_feasible") else "❌")
                        st.metric("N", model["n_obs"])
                    with st.container():
                        csv_url = f"{ds['id']}/regressions/{model['id']}.csv"
                        report_url = f"/api/v1/reports/regression/{model['id']}.csv"
                        st.info(f"CSV export available at: `{report_url}`")
    except Exception as e:
        st.error(f"Failed to load regressions: {e}")

    if st.button("← Back to Datasets"):
        st.switch_page("pages/02_Datasets.py")


dataset_detail()
