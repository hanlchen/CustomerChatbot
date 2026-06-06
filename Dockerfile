# Customer Support Chatbot v2 — Streamlit + OpenAI Agents SDK
FROM python:3.11-slim

# Don't write .pyc files; stream logs unbuffered for container log visibility.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, database, and policy files.
COPY . .

# Streamlit serves on 8501.
EXPOSE 8501

# Basic container healthcheck against Streamlit's health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Bind to all interfaces so the port is reachable from outside the container.
CMD ["streamlit", "run", "chatbot_v2_agents.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
