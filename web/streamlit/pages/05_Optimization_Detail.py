import json

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import get_optimization, list_optimization_points, list_tasks


def optimization_detail():
    if "auth" not in st.session_state:
        st.warning("Please log in first")
        st.page_link("pages/01_Login.py", label="Go to Login")
        return

    opt_id = st.session_state.get("selected_optimization_id")
    if not opt_id:
        st.info("No optimization selected")
        if st.button("← Back to Optimizations"):
            st.switch_page("pages/04_Optimizations.py")
        return

    try:
        opt = get_optimization(opt_id)
    except Exception as e:
        st.error(f"Failed to load optimization: {e}")
        if st.button("← Back"):
            st.switch_page("pages/04_Optimizations.py")
        return

    st.title(f"🎯 {opt.get('name', 'Optimization')}")
    st.caption(f"ID: `{opt['id']}`")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", opt["status"])
    with col2:
        st.metric("Points", opt["n_points"])
    with col3:
        st.metric("Mode", opt["mode"])
    with col4:
        if opt["status"] in ("pending", "processing"):
            if st.button("🔄 Refresh"):
                st.rerun()

    with st.expander("Targets"):
        st.json(opt.get("targets", {}))

    if opt["status"] == "completed":
        st.divider()
        st.subheader("📈 Pareto Frontier")

        try:
            points_data = list_optimization_points(opt_id, page_size=500)
            points = points_data.get("items", [])
        except Exception as e:
            st.error(f"Failed to load points: {e}")
            points = []

        if points:
            df = pd.DataFrame(points)
            df["label"] = df.apply(
                lambda r: "Dominated" if r["is_dominated"] else "Pareto-optimal",
                axis=1,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.scatter(
                    df,
                    x="total_input",
                    y="total_output",
                    color="label",
                    hover_data=["ratio", "efficiency", "inputs", "outputs"],
                    title="Total Input vs Total Output",
                    labels={"total_input": "Total Input", "total_output": "Total Output"},
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig2 = px.scatter(
                    df,
                    x="ratio",
                    y="efficiency",
                    color="label",
                    hover_data=["total_input", "total_output"],
                    title="Efficiency vs Ratio",
                    labels={"ratio": "Ratio", "efficiency": "Efficiency (%)"},
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Pareto Points")
            st.dataframe(
                df[["ratio", "total_input", "total_output", "efficiency", "is_dominated", "rank"]]
                .round(4),
                use_container_width=True,
                hide_index=True,
            )

            pareto_only = df[~df["is_dominated"]]
            st.metric("Pareto-optimal Solutions", len(pareto_only))
            if len(pareto_only) > 0:
                best = pareto_only.loc[pareto_only["efficiency"].idxmin()]
                st.info(f"Best efficiency: {best['efficiency']:.2f}% (ratio={best['ratio']:.4f})")

            csv_url = f"/api/v1/reports/optimization/{opt['id']}.csv"
            st.info(f"📥 CSV export: `{csv_url}`")
        else:
            st.info("No Pareto points generated yet.")
    elif opt["status"] == "failed":
        st.error(f"Optimization failed: {opt.get('error_message', 'Unknown error')}")
    else:
        st.info(f"Optimization is `{opt['status']}`. Waiting for processing...")

        if st.button("🔄 Check Status", use_container_width=True):
            st.rerun()

    if st.button("← Back to Optimizations"):
        st.switch_page("pages/04_Optimizations.py")


optimization_detail()
