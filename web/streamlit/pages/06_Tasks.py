import streamlit as st

from api_client import get_task, list_tasks


def tasks_page():
    if "auth" not in st.session_state:
        st.warning("Please log in first")
        st.page_link("pages/01_Login.py", label="Go to Login")
        return

    st.title("📋 Async Tasks")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    try:
        data = list_tasks()
        items = data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load tasks: {e}")
        return

    if not items:
        st.info("No tasks yet. Upload a dataset or create an optimization to see tasks here.")
        return

    for task in items:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1:
                st.write(f"**{task['task_type']}**")
                st.caption(f"ID: `{task['id']}`")
            with c2:
                status = task["status"]
                color_map = {
                    "PENDING": "🔵",
                    "PROGRESS": "🟡",
                    "SUCCESS": "🟢",
                    "FAILURE": "🔴",
                }
                st.write(f"{color_map.get(status, '⚪')} `{status}`")
            with c3:
                st.progress(task["progress"] / 100, text=f"{task['progress']}%")
            with c4:
                if st.button("🔍 Details", key=f"task_{task['id']}", use_container_width=True):
                    st.session_state.selected_task_id = task["id"]

            if task.get("error_message"):
                st.error(task["error_message"])

    selected_id = st.session_state.get("selected_task_id")
    if selected_id:
        try:
            detail = get_task(selected_id)
            with st.expander(f"Task Detail: {detail['task_type']}", expanded=True):
                st.json(detail)
                if st.button("Close Details"):
                    del st.session_state.selected_task_id
                    st.rerun()
        except Exception as e:
            st.error(f"Failed to load task: {e}")


tasks_page()
