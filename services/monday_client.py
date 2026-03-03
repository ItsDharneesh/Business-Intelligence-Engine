import requests
from config import MONDAY_API_KEY
from utils.trace_logger import log_trace

MONDAY_URL = "https://api.monday.com/v2"
REQUEST_TIMEOUT = 15  # prevent hanging demo


def fetch_board_items(board_id: int):

    if not MONDAY_API_KEY:
        raise ValueError("MONDAY_API_KEY is missing.")

    if not board_id:
        raise ValueError("Board ID is missing.")

    log_trace(f"Calling Monday API for board {board_id}")

    query = f"""
    query {{
    boards(ids: {int(board_id)}) {{
      id
      name

      columns {{
        id
        title
      }}

      items_page(limit: 500) {{
        items {{
          id
          name
          column_values {{
            id
            text
            value
          }}
        }}
      }}
    }}
  }}
  """

    try:
        response = requests.post(
            MONDAY_URL,
            headers={
                "Authorization": MONDAY_API_KEY,
                "Content-Type": "application/json"
            },
            json={"query": query},
            timeout=REQUEST_TIMEOUT
        )
    except requests.exceptions.RequestException as e:
        log_trace(f"Network error calling Monday API: {str(e)}")
        raise Exception("Failed to connect to Monday.com API.")

    if response.status_code != 200:
        log_trace(f"Monday API HTTP error: {response.text}")
        raise Exception(f"Monday API HTTP error: {response.status_code}")

    data = response.json()
    log_trace(f"Monday API raw response keys: {list(data.keys())}")

    boards = data.get("data", {}).get("boards", [])
    log_trace(f"Boards returned: {len(boards)}")

    if boards:
        items = boards[0].get("items_page", {}).get("items", [])
        log_trace(f"Items fetched from board: {len(items)}")
    # Handle GraphQL-level errors
    if "errors" in data:
        log_trace(f"Monday GraphQL error: {data['errors']}")
        raise Exception("Monday API returned GraphQL errors.")

    boards = data.get("data", {}).get("boards", [])

    if not boards:
        log_trace("No boards returned from Monday API.")
        return {"data": {"boards": []}}

    log_trace("Monday API call successful")

    return data