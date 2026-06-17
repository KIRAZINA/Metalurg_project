import json

import streamlit as st

from api_client import (
    create_optimization,
    delete_optimization,
    list_datasets,
    list_optimizations,
)


def optimizations_page():
    if "auth" not in st.session_state:
        st.warning("Please log in first")
        st.page_link("pages/01_Login.py", label="Go to Login")
        return

    st.title("🎯 Pareto Optimizations")

    tab1, tab2 = st.tabs(["My Optimizations", "Create New"])

    with tab1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

        try:
            data = list_optimizations()
            items = data.get("items", [])
            if not items:
                st.info("No optimizations yet. Create one!")
            else:
                for opt in items:
                    with st.expander(f"{opt.get('name', 'Unnamed')}  —  *{opt['status']}*"):
                        c1, c2, c3 = st.columns([2, 1, 1])
                        with c1:
                            st.write(f"**Dataset:** `{opt['dataset_id']}`")
                            st.write(f"**Status:** `{opt['status']}`")
                            st.write(f"**Points:** {opt['n_points']}  |  **Mode:** {opt['mode']}")
                            st.write(f"**Targets:** {json.dumps(opt.get('targets', {}), indent=2)}")
                            if opt.get("error_message"):
                                st.error(opt["error_message"])
                        with c2:
                            if st.button("🔍 Open", key=f"open_opt_{opt['id']}", use_container_width=True):
                                st.session_state.selected_optimization_id = opt["id"]
                                st.switch_page("pages/05_Optimization_Detail.py")
                        with c3:
                            if st.button("🗑️ Delete", key=f"del_opt_{opt['id']}", use_container_width=True):
                                try:
                                    delete_optimization(opt["id"])
                                    st.success("Deleted")
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))
        except Exception as e:
            st.error(f"Failed to load optimizations: {e}")

    with tab2:
        st.subheader("Create Optimization")

        try:
            ds_data = list_datasets(page_size=100)
            ready_datasets = [
                ds for ds in ds_data.get("items", [])
                if ds["status"] == "ready"
            ]
        except Exception as e:
            st.error(f"Failed to load datasets: {e}")
            ready_datasets = []

        if not ready_datasets:
            st.warning("No ready datasets available. Upload and process a dataset first.")
            return

        ds_options = {ds["name"]: ds["id"] for ds in ready_datasets}
        selected_name = st.selectbox("Dataset", list(ds_options.keys()))
        ds_id = ds_options[selected_name]
        opt_name = st.text_input("Optimization Name (optional)")
        n_points = st.slider("Number of Points", 10, 500, 100)

        st.write("**Target Configuration**")
        st.caption("Define element targets as JSON: `{\"Element\": {\"x_column\": \"ColName\", \"target_value\": 0.5}}`")
        targets_str = st.text_area(
            "Targets (JSON)",
            value='{\n  "Fe": {"x_column": "X1", "target_value": 90},\n  "C": {"x_column": "X2", "target_value": 0.5}\n}',
            height=150,
        )

        if st.button("🚀 Create & Run", type="primary", use_container_width=True):
            try:
                targets = json.loads(targets_str)
                result = create_optimization(
                    dataset_id=ds_id,
                    targets=targets,
                    name=opt_name or None,
                    n_points=n_points,
                )
                st.success(f"Optimization created! ID: `{result['id']}`")
                st.session_state.selected_optimization_id = result["id"]
                st.switch_page("pages/05_Optimization_Detail.py")
            except json.JSONDecodeError:
                st.error("Invalid JSON in targets")
            except Exception as e:
                st.error(f"Failed to create optimization: {e}")


optimizations_page()
