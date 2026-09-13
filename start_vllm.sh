#!/usr/bin/env bash
#
# Start a local vLLM server for CustomerChatbot.
#
#   ./start_vllm.sh                    # default model
#   ./start_vllm.sh Qwen/Qwen3-4B      # pick a model
#
# Then, in another terminal, run the app. The settled way is a .env file next
# to app.py, so every terminal agrees and nothing has to be exported:
#
#   CHATBOT_BASE_URL=http://localhost:8000/v1
#   CHATBOT_MODEL=<the same model id you passed here>
#   API_PORT=8001                      # vLLM already holds 8000
#
# then just:  python app.py
#
# (app.py has no --port flag; the port comes from API_PORT.)
#
# Tool calling REQUIRES both --enable-auto-tool-choice and a --tool-call-parser
# matching the model family. Without them vLLM returns tool calls as plain
# text, the agent sees no tool_calls, and every answer becomes a guess.

set -euo pipefail

MODEL="${1:-${CHATBOT_MODEL:-Qwen/Qwen3-8B}}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-16384}"

# --- pick the tool-call parser from the model family ------------------------
# Qwen and Hermes models use Hermes-style tool syntax; Llama 3.x emits JSON;
# Mistral has its own [TOOL_CALLS] format.
lower_model="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
case "$lower_model" in
  *qwen*|*hermes*)   PARSER="hermes" ;;
  *llama-3.1*|*llama-3.2*|*llama-3.3*|*llama3*) PARSER="llama3_json" ;;
  *llama-4*|*llama4*) PARSER="llama4_pythonic" ;;
  *mistral*|*devstral*) PARSER="mistral" ;;
  *deepseek*)        PARSER="deepseek_v3" ;;
  *)
    echo "!! Unknown model family: $MODEL"
    echo "!! Set VLLM_TOOL_PARSER explicitly. See:"
    echo "!! https://docs.vllm.ai/en/latest/features/tool_calling/"
    PARSER="hermes"
    ;;
esac
PARSER="${VLLM_TOOL_PARSER:-$PARSER}"

echo "model:   $MODEL"
echo "parser:  $PARSER"
echo "address: http://${HOST}:${PORT}/v1"
echo

# --- Apple Silicon needs the metal plugin ----------------------------------
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "Apple Silicon detected."

  # Activate the Metal plugin environment automatically if it exists and
  # isn't already active, so this works without remembering to source it.
  METAL_VENV="${VLLM_METAL_VENV:-$HOME/.venv-vllm-metal}"
  if ! python -c "import vllm" 2>/dev/null && [[ -f "$METAL_VENV/bin/activate" ]]; then
    echo "Activating $METAL_VENV"
    # shellcheck disable=SC1091
    source "$METAL_VENV/bin/activate"
  fi

  if ! python -c "import vllm" 2>/dev/null; then
    cat <<'SETUP'
vLLM is not installed. On Apple Silicon it runs through the Metal plugin:

    curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
    source ~/.venv-vllm-metal/bin/activate

Requires native arm64 Python 3.12 (not Rosetta) and Xcode Command Line Tools.
Use MLX-format weights, e.g. mlx-community/Qwen3-8B-4bit.

Alternative, if you already run Docker Desktop 4.62+:

    docker model install-runner --backend vllm
    docker model pull mlx-community/Qwen3-8B-4bit
    docker model run mlx-community/Qwen3-8B-4bit

SETUP
    exit 1
  fi
fi

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --enable-auto-tool-choice \
  --tool-call-parser "$PARSER" \
  --enable-prefix-caching \
  "${@:2}"
