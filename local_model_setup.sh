#!/usr/bin/env bash
#
# Set up a local open-weights model for CustomerChatbot on Apple Silicon.
#
#   ./local_model_setup.sh              # check the machine, report what's needed
#   ./local_model_setup.sh --install    # install vLLM via the Metal plugin
#   ./local_model_setup.sh --docker     # use Docker Model Runner instead
#
# Runs checks first and stops at the first real blocker, so you find out about
# a wrong Python before waiting on a 5 GB download.

set -uo pipefail

MODEL="${MODEL:-mlx-community/Qwen3-8B-4bit}"
VENV="$HOME/.venv-vllm-metal"
MODE="check"

for arg in "$@"; do
  case "$arg" in
    --install) MODE="install" ;;
    --docker)  MODE="docker" ;;
    --model=*) MODEL="${arg#*=}" ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

# --- output helpers ---------------------------------------------------------
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'
  GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""
fi
ok()   { echo "  ${GREEN}✓${RESET} $1"; }
bad()  { echo "  ${RED}✗${RESET} $1"; }
warn() { echo "  ${YELLOW}!${RESET} $1"; }
note() { echo "    ${DIM}$1${RESET}"; }
head_() { echo; echo "${BOLD}$1${RESET}"; }

BLOCKERS=0

# ---------------------------------------------------------------------------
head_ "Machine"
# ---------------------------------------------------------------------------

if [[ "$(uname -s)" != "Darwin" ]]; then
  bad "Not macOS. This script sets up the Apple Silicon path."
  note "On Linux with an NVIDIA GPU: pip install vllm, then ./start_vllm.sh"
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  bad "Not Apple Silicon (found $(uname -m))."
  note "The Metal backend needs an M-series chip."
  exit 1
fi
ok "macOS on Apple Silicon"

RAM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
RAM_GB=$(( RAM_BYTES / 1024 / 1024 / 1024 ))
CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
ok "$CHIP · ${RAM_GB} GB unified memory"

# Rough fit guidance. macOS itself needs ~4-5 GB.
if   (( RAM_GB >= 32 )); then note "Fits up to ~30B at 4-bit."
elif (( RAM_GB >= 24 )); then note "Fits up to ~14B at 4-bit."
elif (( RAM_GB >= 16 )); then note "Fits up to ~9B at 4-bit. Qwen3-8B-4bit is the sweet spot."
elif (( RAM_GB >= 8 ));  then
  warn "8 GB is tight — use a 3-4B model, e.g. mlx-community/Qwen3-4B-4bit"
  note "An 8B model will swap to disk and crawl."
fi

# ---------------------------------------------------------------------------
if [[ "$MODE" == "docker" ]]; then
  head_ "Docker Model Runner path"

  if ! command -v docker >/dev/null 2>&1; then
    bad "docker not found. Install Docker Desktop 4.62 or later."
    exit 1
  fi
  ok "docker present ($(docker --version 2>/dev/null | head -1))"

  echo
  echo "Run these:"
  echo "  ${BOLD}docker model install-runner --backend vllm${RESET}"
  echo "  ${BOLD}docker model pull $MODEL${RESET}"
  echo "  ${BOLD}docker model run $MODEL${RESET}"
  echo
  echo "Then point the app at it (check the port Docker prints):"
  echo "  export CHATBOT_BASE_URL=http://localhost:12434/engines/v1"
  echo "  export CHATBOT_MODEL=$MODEL"
  echo
  note "Docker Model Runner handles tool-call parsing itself, so the"
  note "--tool-call-parser flag in start_vllm.sh does not apply here."
  exit 0
fi

# ---------------------------------------------------------------------------
head_ "Prerequisites for the Metal plugin"
# ---------------------------------------------------------------------------

if xcode-select -p >/dev/null 2>&1; then
  ok "Xcode Command Line Tools installed"
else
  bad "Xcode Command Line Tools missing"
  note "Fix: xcode-select --install"
  BLOCKERS=$((BLOCKERS+1))
fi

# vllm-metal requires a NATIVE arm64 Python 3.12. A Rosetta/x86_64 Python
# (common with older conda installs) fails in confusing ways.
PY312=""
CANDIDATES=(python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 /usr/bin/python3.12)

# Also look through conda environments -- if conda is already installed, making
# a 3.12 env there is easier than adding another package manager.
if command -v conda >/dev/null 2>&1; then
  CONDA_ROOT="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_ROOT" && -d "$CONDA_ROOT/envs" ]]; then
    while IFS= read -r found; do
      CANDIDATES+=("$found")
    done < <(find "$CONDA_ROOT/envs" -maxdepth 3 -name 'python3.12' -type f 2>/dev/null)
  fi
fi

for candidate in "${CANDIDATES[@]}"; do
  resolved="$(command -v "$candidate" 2>/dev/null || { [[ -x "$candidate" ]] && echo "$candidate"; })"
  [[ -z "$resolved" ]] && continue
  arch_out="$("$resolved" -c 'import platform;print(platform.machine())' 2>/dev/null)"
  if [[ "$arch_out" == "arm64" ]]; then PY312="$resolved"; break; fi
done

if [[ -n "$PY312" ]]; then
  ok "native arm64 Python 3.12 at $PY312"
else
  bad "no native arm64 Python 3.12 found"
  # Recommend whichever tool they already have.
  if command -v conda >/dev/null 2>&1; then
    note "You already have conda, so this is the shortest path:"
    note "    conda create -n vllm-metal python=3.12 -y"
    note "    conda activate vllm-metal"
    note "    ./local_model_setup.sh --install"
  elif command -v brew >/dev/null 2>&1; then
    note "Fix: brew install python@3.12"
  else
    note "Install one of:"
    note "    conda create -n vllm-metal python=3.12 -y   (if you use conda)"
    note "    brew install python@3.12                    (needs Homebrew)"
  fi
  note "A Rosetta/x86_64 Python will not work with the Metal backend."
  BLOCKERS=$((BLOCKERS+1))
fi

CURRENT_PY="$(command -v python3 || true)"
if [[ -n "$CURRENT_PY" ]]; then
  cur_ver="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo '?')"
  cur_arch="$(python3 -c 'import platform;print(platform.machine())' 2>/dev/null || echo '?')"
  note "your default python3: $CURRENT_PY ($cur_ver, $cur_arch)"
  note "the app can stay on this one — vLLM runs in its own environment"
fi

# ---------------------------------------------------------------------------
head_ "vLLM"
# ---------------------------------------------------------------------------

if [[ -d "$VENV" ]]; then
  ok "Metal plugin environment exists at $VENV"
  INSTALLED=1
else
  warn "not installed yet"
  INSTALLED=0
fi

if (( BLOCKERS > 0 )); then
  echo
  echo "${RED}${BOLD}Fix the $BLOCKERS item(s) above, then re-run.${RESET}"
  exit 1
fi

if [[ "$MODE" == "install" && "$INSTALLED" == "0" ]]; then
  head_ "Installing"
  echo "  This downloads vLLM and the Metal plugin into $VENV"
  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash || {
    bad "install failed"
    note "See https://github.com/vllm-project/vllm-metal for manual steps."
    exit 1
  }
  ok "installed"
  INSTALLED=1
fi

# ---------------------------------------------------------------------------
head_ "Next steps"
# ---------------------------------------------------------------------------

if (( INSTALLED == 0 )); then
  echo "  Install vLLM:"
  echo "    ${BOLD}./local_model_setup.sh --install${RESET}"
  echo
  echo "  Or, if you'd rather use Docker Desktop:"
  echo "    ${BOLD}./local_model_setup.sh --docker${RESET}"
  exit 0
fi

cat <<EOF

  ${BOLD}1. Serve the model${RESET}  (terminal 1 — first run downloads ~5 GB)

     source $VENV/bin/activate
     ./start_vllm.sh $MODEL

     Wait for: "Application startup complete"

  ${BOLD}2. Point the app at it${RESET}  (terminal 2 — your normal Python is fine)

     export CHATBOT_BASE_URL=http://localhost:8000/v1
     export CHATBOT_MODEL=$MODEL
     API_PORT=8001 python app.py

     vLLM takes port 8000, so the app moves to 8001.

  ${BOLD}3. Check the model can actually use your tools${RESET}

     python trace_turn.py "where is my order"

     This prints every prompt, tool call and result for one turn. If the
     model writes tool syntax as plain text instead of calling anything,
     the launch flags are wrong -- see --enable-auto-tool-choice above.

  ${BOLD}4. Use it${RESET}

     open http://localhost:8001/ui
     curl -s localhost:8001/status | python -m json.tool   # confirms the model

EOF
