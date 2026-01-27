# Customer Support Chatbot v2

A policy-aware customer support chatbot built with OpenAI Agents SDK, featuring multi-agent orchestration, RAG-based policy search, and real-time metrics tracking.

## Features

- **Multi-Agent Architecture**: Specialized agents for different tasks
  - **Support Agent**: Main orchestrator with intent classification
  - **Data Agent**: Handles order queries and database lookups
  - **Policy Agent**: Answers policy questions using RAG

- **Intent Classification**: Automatically routes messages to the right agent
  - GREETING, POLICY_QUESTION, DATA_QUESTION, ACTION_REQUEST, OTHER

- **Policy RAG Search**: Fast keyword-based policy lookup from local cache

- **Security**:
  - Two-factor authentication (email + phone)
  - SQL injection prevention
  - Customer data isolation
  - Email address protection

- **Performance**:
  - Response caching for policy questions (5-min TTL)
  - Rate limiting (20 requests/minute)
  - Fast local policy search (no API calls)

- **Metrics Tracking**:
  - Token usage (prompt + completion)
  - Request latency
  - Cache hit rate
  - Intent/policy/data query counts

## Prerequisites

- Python 3.10+
- OpenAI API key
- Pinecone API key (optional, for vector search)

## Setup

1. **Clone and navigate to the project**:
   ```bash
   cd chatpotwithsql
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** in `.env`:
   ```
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX_NAME=policy-chatbot
   ENABLE_TRACING=0
   ```

## Running the App

```bash
streamlit run chatbot_v2_agents.py
```

The app will open at `http://localhost:8501`

## Test Accounts

| Email | Phone |
|-------|-------|
| alice@example.com | 555-0101 |
| bob@example.com | 555-0102 |
| carol@example.com | 555-0103 |

## Sample Questions

**Data Questions** (routed to Data Agent):
- "What are my orders?"
- "What did I order last month?"
- "Show me order 101"

**Policy Questions** (routed to Policy Agent):
- "What's your return policy?"
- "Can I cancel an order?"
- "How do refunds work?"

**Action Requests** (routed to Policy Agent):
- "Cancel order 101"
- "I want a refund"

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Message                       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Support Agent (Orchestrator)            │
│  • Classifies intent using classify_intent tool     │
│  • Routes to specialized agents via handoffs        │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│   Data    │  │  Policy   │  │  Greeting │
│   Agent   │  │   Agent   │  │ (handled  │
│           │  │           │  │  inline)  │
└─────┬─────┘  └─────┬─────┘  └───────────┘
      │              │
      ▼              ▼
┌───────────┐  ┌───────────┐
│  SQLite   │  │  Policy   │
│  Database │  │   Cache   │
└───────────┘  └───────────┘
```

## File Structure

```
chatpotwithsql/
├── chatbot_v2_agents.py   # Main application
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not in git)
├── README.md              # This file
├── database/
│   ├── __init__.py
│   └── chatbot.db         # SQLite database
├── policies/
│   ├── __init__.py
│   └── documents.py       # Policy documents
└── others/                # Legacy files (LangChain version, evals, etc.)
```

## Metrics Dashboard

The sidebar displays real-time metrics:

| Metric | Description |
|--------|-------------|
| Total Tokens | Cumulative token usage |
| Requests | Number of API calls |
| Avg Latency | Average response time |
| Cache Rate | Cache hit percentage |

Click "Detailed Metrics" for full breakdown including:
- Intent classification count
- Policy search count
- Data query count
- Error count

## Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `PINECONE_API_KEY` | optional | Pinecone API key for vector search |
| `PINECONE_INDEX_NAME` | policy-chatbot | Pinecone index name |
| `ENABLE_TRACING` | 0 | Set to 1 to enable OpenAI tracing |

## OpenAI Tracing

To enable OpenAI's built-in tracing:

1. Set `ENABLE_TRACING=1` in `.env`
2. View traces at [platform.openai.com/traces](https://platform.openai.com/traces)

## License

MIT
