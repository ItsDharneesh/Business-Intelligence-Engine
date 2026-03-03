import streamlit as st


def init_trace():
    """
    Initialize trace list in session state.
    """
    if "trace" not in st.session_state:
        st.session_state["trace"] = []


def reset_trace():
    """
    Clear previous trace before new query.
    """
    st.session_state["trace"] = []


def log_trace(message: str):
    """
    Append a message to tool execution trace.
    """
    if "trace" not in st.session_state:
        init_trace()

    # Convert everything safely to string
    st.session_state["trace"].append(str(message))


def get_trace():
    """
    Return current trace list.
    """
    return st.session_state.get("trace", [])