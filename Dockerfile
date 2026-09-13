# CustomerChatbot — FastAPI support agent
#
# Runs the API on :8000. The chat UI (chat_interface.html) is a static file
# served from the same origin at /ui.
#
#   docker build -t customerchatbot .
#   docker run -p 8000:8000 customerchatbot
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... customerchatbot

FROM python:3.11-slim

# Don't write .pyc files; stream logs unbuffered for container log visibility.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, data and policies.
COPY . .

# Run as a non-root user. The app writes nothing to disk, so read-only
# ownership of /app is sufficient.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# /health reports process liveness; /status additionally reports which
# reasoning engine and retrieval backend resolved at startup.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

# Bind to all interfaces so the port is reachable from outside the container.
# One worker: sessions live in process memory (see README, "Known limits").
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
