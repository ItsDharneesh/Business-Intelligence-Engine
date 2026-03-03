from dotenv import load_dotenv
import os

load_dotenv()

MONDAY_API_KEY = os.getenv("MONDAY_API_KEY")
DEALS_BOARD_ID = os.getenv("DEALS_BOARD_ID")
WORK_ORDERS_BOARD_ID = os.getenv("WORK_ORDERS_BOARD_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")