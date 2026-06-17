import streamlit as st

from api_client import delete_dataset, list_datasets, upload_dataset


def datasets_page():
    if "auth" not in st.session_state:
        st.warning("Please log in first")
        st.page_link("pages/01_Login.py", label="Go to Login")
        return

    st.title("📊 Datasets")

    tab1, tab2 = st.tabs(["My Datasets", "Upload New"])

    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Your Datasets")
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        try:
            data = list_datasets()
            items = data.get("items", [])
            if not items:
                st.info("No datasets yet. Upload one!")
            else:
                for ds in items:
                    with st.expander(f"{ds['name']}  —  *{ds['status']}*"):
                        c1, c2, c3 = st.columns([2, 1, 1])
                        with c1:
                            st.write(f"**File:** {ds.get('original_filename', 'N/A')}")
                            st.write(f"**Size:** {ds.get('file_size_bytes', 0):,} bytes")
                            st.write(f"**Rows:** {ds.get('row_count', '—')}  |  **Cols:** {ds.get('column_count', '—')}")
                            st.write(f"**Status:** `{ds['status']}`")
                            if ds.get("error_message"):
                                st.error(ds["error_message"])
                        with c2:
                            if st.button("🔍 Open", key=f"open_{ds['id']}", use_container_width=True):
                                st.session_state.selected_dataset_id = ds["id"]
                                st.switch_page("pages/03_Dataset_Detail.py")
                        with c3:
                            if st.button("🗑️ Delete", key=f"del_{ds['id']}", use_container_width=True):
                                try:
                                    delete_dataset(ds["id"])
                                    st.success("Deleted")
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))
        except Exception as e:
            st.error(f"Failed to load datasets: {e}")

    with tab2:
        st.subheader("Upload Excel File")
        uploaded_file = st.file_uploader("Choose an .xls or .xlsx file", type=["xls", "xlsx"])
        ds_name = st.text_input("Dataset Name")
        ds_desc = st.text_area("Description (optional)")

        if uploaded_file and ds_name and st.button("📤 Upload", type="primary", use_container_width=True):
            try:
                result = upload_dataset(uploaded_file, ds_name, ds_desc or None)
                st.success(f"Uploaded: {result['name']}")
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")


datasets_page()
