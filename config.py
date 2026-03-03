import os
import streamlit as st

def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
MONDAY_API_KEY = get_secret("MONDAY_API_KEY")
DEALS_BOARD_ID = get_secret("DEALS_BOARD_ID")
WORK_ORDERS_BOARD_ID = get_secret("WORK_ORDERS_BOARD_ID")
