import streamlit as st
from agent.agent_core import run_agent
from utils.trace_logger import init_trace, reset_trace, get_trace

# ---------------------------------------
# Page Config
# ---------------------------------------
st.set_page_config(page_title="Monday BI Agent", layout="wide")

st.title("📊 Monday.com Business Intelligence Agent")

# Initialize trace memory
init_trace()

# ---------------------------------------
# Input
# ---------------------------------------
query = st.text_input("Ask a founder-level question")

# ---------------------------------------
# Execution
# ---------------------------------------
if st.button("Submit") and query:

    # Reset logs for this run
    reset_trace()

    with st.spinner("Analyzing business data..."):
        response = run_agent(query)

    # ---------------------------------------
    # Show Executive Response
    # ---------------------------------------
    st.subheader("📈 Executive Insight")
    st.write(response)

    # ---------------------------------------
    # Show Debug Trace
    # ---------------------------------------
    st.subheader("🔧 Tool Execution Trace")

    trace_logs = get_trace()

    if trace_logs:
        for step in trace_logs:
            st.write(f"• {step}")
    else:
        st.warning("No trace logs captured.")