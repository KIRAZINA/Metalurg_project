import streamlit as st

from api_client import AuthState, login, register


def show_login():
    st.title("Test Metal Dashboard")

    tab1, tab2 = st.tabs(["Sign In", "Register"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            if submitted:
                if not email or not password:
                    st.error("Email and password are required")
                else:
                    try:
                        data = login(email, password)
                        st.session_state.auth = AuthState(
                            access_token=data["access_token"],
                            refresh_token=data["refresh_token"],
                        )
                        st.session_state.user = data.get("user", {"email": email})
                        st.success("Logged in successfully")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {e}")

    with tab2:
        with st.form("register_form"):
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_name = st.text_input("Full Name (optional)")
            submitted = st.form_submit_button("Register", use_container_width=True)
            if submitted:
                if not reg_email or not reg_password:
                    st.error("Email and password are required")
                elif len(reg_password) < 8:
                    st.error("Password must be at least 8 characters")
                else:
                    try:
                        register(reg_email, reg_password, reg_name or None)
                        st.success("Registration successful! Please sign in.")
                    except Exception as e:
                        st.error(f"Registration failed: {e}")


if "auth" not in st.session_state:
    show_login()
else:
    st.success(f"Logged in as {st.session_state.get('user', {}).get('email', 'user')}")
    st.page_link("app.py", label="Go to Dashboard", use_container_width=True)
