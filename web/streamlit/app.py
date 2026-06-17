import streamlit as st

from config import APP_ICON, APP_TITLE

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.divider()

        if "auth" in st.session_state:
            user = st.session_state.get("user", {})
            st.write(f"👤 {user.get('email', 'User')}")
            if st.button("🚪 Logout", use_container_width=True, type="primary"):
                del st.session_state.auth
                if "user" in st.session_state:
                    del st.session_state.user
                st.rerun()
            st.divider()

            st.page_link("app.py", label="🏠 Home", use_container_width=True)
            st.page_link("pages/02_Datasets.py", label="📊 Datasets", use_container_width=True)
            st.page_link("pages/03_Dataset_Detail.py", label="📂 Dataset Detail", use_container_width=True)
            st.page_link("pages/04_Optimizations.py", label="🎯 Optimizations", use_container_width=True)
            st.page_link("pages/05_Optimization_Detail.py", label="📈 Optimization Detail", use_container_width=True)
            st.page_link("pages/06_Tasks.py", label="📋 Tasks", use_container_width=True)
        else:
            st.page_link("pages/01_Login.py", label="🔑 Login / Register", use_container_width=True)

    if "auth" in st.session_state:
        st.title("🏠 Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        try:
            from api_client import list_datasets, list_optimizations, list_tasks

            ds_data = list_datasets(page_size=1)
            opt_data = list_optimizations(page_size=1)
            task_data = list_tasks(page_size=1)

            with col1:
                st.metric("Datasets", ds_data.get("total", 0))
            with col2:
                st.metric("Optimizations", opt_data.get("total", 0))
            with col3:
                st.metric("Tasks", task_data.get("total", 0))
            with col4:
                running = sum(
                    1 for t in task_data.get("items", [])
                    if t.get("status") in ("PENDING", "PROGRESS")
                )
                st.metric("Running Tasks", running)
        except Exception as e:
            st.warning(f"Could not load stats: {e}")

        st.divider()
        st.markdown("""
        ### Quick Actions
        - **📊 [Upload Dataset](pages/02_Datasets)** — Upload an Excel file for analysis
        - **🎯 [Create Optimization](pages/04_Optimizations)** — Run Pareto optimization on processed data
        - **📋 [View Tasks](pages/06_Tasks)** — Check status of background tasks
        """)
    else:
        st.title(f"Welcome to {APP_TITLE}")
        st.markdown("""
        A web application for linear regression analysis and Pareto optimization
        of physicochemical properties of steel.

        **Get started by logging in or creating an account.**
        """)
        st.page_link("pages/01_Login.py", label="🔑 Login / Register", use_container_width=True)


if __name__ == "__main__":
    main()
