# 📊 Business Intelligence Engine — LLM-Powered Monday.com Analytics

> An AI-driven Business Intelligence Agent that converts natural language into deterministic, live business insights from Monday.com boards.

---

## 🚀 Project Overview

The **Business Intelligence Engine** is a production-ready AI agent that:

- Connects to **live Monday.com boards**
- Uses **GPT-5.2 tool-calling** to interpret founder-level questions
- Executes **deterministic pandas analytics**
- Prevents hallucinated numbers
- Displays full execution trace for transparency

It allows users to ask:

- “How’s our pipeline looking?”
- “Which sector is underperforming?”
- “Are there closed deals without work orders?”
- “Where are we leaking revenue?”

All answers are grounded in real data.

---

## 🏛️ Architecture

```
User Query (Streamlit UI)
        ↓
GPT-4o (Tool Selection)
        ↓
Tool Registry (Python Backend)
        ↓
Monday.com Live API
        ↓
Data Normalization
        ↓
Deterministic BI Engine (pandas)
        ↓
Grounded Executive Summary
        ↓
Full Trace Output
```

---

## 📁 Repository Structure

```
Business-Intelligence-Engine/
│
├── app.py
├── config.py
├── agent/
│   └── agent_core.py
│
├── services/
│   └── monday_client.py
│
├── utils/
│   ├── normalization.py
│   ├── trace_logger.py
│   └── bi_engine.py
│
├── tests/
├── requirements.txt
└── README.md
```

---

## 📌 Core Components

### 🧠 `agent_core.py`
- Handles GPT-5.2 tool selection
- Sends tool schema to LLM
- Executes selected backend tool
- Sends results back for grounded summary
- Enforces no-hallucination rule

---

### 📡 `monday_client.py`
- Executes live GraphQL queries
- Fetches board items
- No caching
- Real-time data pull per query

---

### 🧹 `normalization.py`
- Maps raw Monday column IDs to clean schema
- Parses JSON values
- Cleans currency & numbers
- Handles nulls
- Standardizes dates

---

### 📊 `bi_engine.py`
Deterministic analytics layer.

Includes:
- Sector filtering
- Quarter filtering
- Pipeline value calculation
- Conversion rate
- Owner performance
- Cross-board validation
- Delayed revenue analysis

All metrics computed via pandas — never by the LLM.

---

### 🔍 `trace_logger.py`
- Logs tool selection
- Logs API calls
- Logs data shapes after filtering
- Logs computed metrics
- Displays full execution trace in UI

---

## ⚙️ How It Works

1. User asks a question.
2. GPT-5.2 selects the correct tool.
3. Backend fetches live data.
4. Data is normalized and filtered.
5. Deterministic metrics are computed.
6. Results are returned to GPT-4o.
7. GPT-5.2 generates a grounded executive summary.
8. Full trace is displayed.

---

## 🛠️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/ItsDharneesh/Business-Intelligence-Engine.git
cd Business-Intelligence-Engine
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key
MONDAY_API_KEY=your_monday_key
DEALS_BOARD_ID=your_deals_board_id
WORK_ORDERS_BOARD_ID=your_work_orders_board_id
```

---

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🧪 Example Test Questions

Try asking:

- “What is our total pipeline value?”
- “Average deal size per owner?”
- “Conversion rate this quarter?”
- “How is the energy sector performing?”
- “Are there closed deals without work orders?”
- “How much revenue is delayed?”

---

## 🎯 Design Principles

### ✅ Zero Hallucination
LLM never generates metrics independently.

### ✅ Deterministic Analytics
All numbers computed via pandas.

### ✅ Live Data Only
Every query hits Monday.com API.

### ✅ Transparent Debugging
Full tool execution trace visible.

### ✅ Production-Ready Architecture
Modular, scalable, extensible.

---

## 🔮 Future Improvements

- Smarter fuzzy matching for sectors
- Multi-tool aggregation queries
- Authentication & role-based access
- Dockerized deployment
- Advanced caching strategies

---

## 📜 License

MIT License

---

## ⭐ Final Note

This project demonstrates:

- Real-world API integration
- LLM tool-calling architecture
- Deterministic BI computation
- Cross-board intelligence
- Production-grade traceability

A foundation for building AI-powered internal analytics systems.
