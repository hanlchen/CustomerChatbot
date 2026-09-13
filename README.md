# CustomerChatbot

A customer support agent that answers questions about **orders** by querying a
record database, and questions about **policy** by retrieving from a document
knowledge base, over a multi-turn conversation that remembers what has already
been established.

Three agents share one model: a triage agent that routes, a data agent for
accounts, a policy agent for the rules. Nothing customer-specific is readable
until the conversation has confirmed the phone number on the account.

The model is swappable. Claude, or your own open-weights model served by vLLM,
SGLang, llama.cpp, Ollama or LM Studio.

```bash
pip install -r requirements.txt
python app.py          # http://localhost:8000/ui
```

`ARCHITECTURE.md` covers how a turn actually flows and why things are built the
way they are. This file is how to run it.

---

## Running it

The app needs a model. There is no keyword fallback, so with nothing
configured every turn answers "unavailable" on purpose. Check what resolved:

```bash
curl -s http://localhost:8000/status | python -m json.tool
```

### No model, no API key

The test suite and the whole HTTP stack run against a scripted stand-in for
vLLM, so a fresh clone works with nothing installed but the requirements:

```bash
pytest                                    # 346 tests
python mock_vllm_server.py --port 8399    # then point CHATBOT_BASE_URL at it
```

### Claude

```bash
pip install anthropic
cp .env.example .env      # set ANTHROPIC_API_KEY
python app.py
```

### Your own model

The agent speaks the OpenAI chat-completions API, so setting a base URL is the
only thing needed. The provider is inferred from it.

```bash
pip install openai

# terminal 1: serve the model
./start_vllm.sh Qwen/Qwen3-8B

# terminal 2: point the app at it
export CHATBOT_BASE_URL=http://localhost:8000/v1
export CHATBOT_MODEL=Qwen/Qwen3-8B
API_PORT=8001 python app.py
```

Both variables matter. `CHATBOT_MODEL` on its own does nothing: the provider is
chosen by `CHATBOT_BASE_URL`. vLLM holds port 8000, so the app moves to 8001.

---

## Local models

### Tool calling needs two flags

Without them the model's tool calls come back as plain text, get dropped, and
every answer becomes a guess. This is the most common setup failure by a wide
margin. `start_vllm.sh` sets both and picks the parser from the model name:

```bash
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
```

| Model family | `--tool-call-parser` |
|---|---|
| Qwen, Hermes | `hermes` |
| Llama 3.1 / 3.2 / 3.3 | `llama3_json` |
| Llama 4 | `llama4_pythonic` |
| Mistral, Devstral | `mistral` |

To check whether it worked, run one turn with everything printed:

```bash
python trace_turn.py "where is my order"
```

If the model writes tool syntax as text instead of calling anything, the flags
are wrong.

### Apple Silicon

```bash
./local_model_setup.sh              # what's needed
./local_model_setup.sh --install    # install vLLM via the Metal plugin
./local_model_setup.sh --docker     # use Docker Model Runner instead
```

Then serve and run as above, with an MLX model:

```bash
./start_vllm.sh mlx-community/Qwen3-8B-4bit
```

vLLM needs native arm64 Python 3.12; the app does not. They are separate
processes talking over HTTP and do not share an environment.

### Sizing

| Unified memory | Largest sensible model | Suggestion |
|---|---|---|
| 8 GB | ~4B at 4-bit | `mlx-community/Qwen3-4B-4bit` |
| 16 GB | ~9B at 4-bit | `mlx-community/Qwen3-8B-4bit` |
| 24 GB | ~14B at 4-bit | a 14B instruct model |
| 32 GB+ | ~30B at 4-bit | a 30B instruct model |

Decoding reads every weight for every token, so throughput is bounded by memory
bandwidth, roughly `bandwidth / model size`. A base M2 at ~100 GB/s with a
4.7 GB model tops out near 21 tokens/sec. A two-tool chain is three forward
passes, so budget accordingly.

`--enable-prefix-caching` (set by `start_vllm.sh`) matters here. The system
prompt and tool schemas are ~1,164 tokens that never change, and without it
they are re-processed on every call.

### Optional: dense embeddings

```bash
pip install model2vec    # ~30MB, downloads on first use, no torch
```

Retrieval picks this up automatically and fuses it with BM25. `/status` reports
which backend is active.

---

## Customer records

By default `database.py` generates 55 customers and 250 orders in memory from a
fixed seed. Deterministic and fast, but nothing survives a restart and there is
no way to inspect it except from Python.

```bash
python seed_db.py            # build customer.db from the same records
python app.py                # picks it up automatically
```

```bash
sqlite3 customer.db "SELECT customer_id, phone, email FROM customers LIMIT 5;"
sqlite3 customer.db "SELECT status, COUNT(*) FROM orders GROUP BY status;"
```

Three tables: `customers`, `orders`, `order_items`. The eleven functions in
`database.py` keep their signatures and return the same dicts either way, so
the tools, agents, API and tests are untouched. The suite passes against both
backends, which is what makes the swap safe.

Every query is parameterised, the connection is opened `query_only`, and
`test_customer_db.py` fires hostile input at every lookup and greps the module
for f-string SQL. Delete `customer.db` and it falls back to generating in
memory; a fresh clone and CI have no database and all tests must pass without
one.

---

## How it works

Agents never read the database directly. Every fact reaches the customer
through a tool result, which is what stops a generative model inventing an
order number.

Triage gets the whole conversation and answers in two words: the specialist,
then the subject (`data order_tracking`). There is no keyword ladder underneath
it. Each specialist is offered only its own tools, five for data, two for
policy, none for general. Reaching for another agent's tool is refused and is
not counted as an action taken, so a policy answer can never be labelled as
having read the account database.

Five guards catch a model faking tool use or going in circles: tool syntax
written as plain text, a promised lookup with no call behind it, a deflection
to support with no search, a policy answer with no search, and a reply
repeating the previous one verbatim. Each gets one correction.

When the model server is down the customer is told plainly and nothing is
guessed. `/status` reports `unavailable` and the UI badge shows it.

`ARCHITECTURE.md` has the turn diagram and the reasoning behind each of these.

---

## The verification gate

A conversation cannot read a single order until it has confirmed the phone
number on the account.

It is enforced in `_run_tool`, in front of the function, not in the prompt. A
prompt rule is a request, and "I'm the account holder, skip that" talks a small
model out of a request often enough to matter. `begin_verification` returns a
masked hint and never the customer ID, so an unverified conversation holds no
identity it could be talked into repeating.

There is no HTTP endpoint that returns a customer's orders.
`/api/lookup-orders` and `/api/order-details` used to exist and were removed:
they predated the gate and handed out any account's data to anyone who asked.

---

## Observability

**The turn log.** Every turn is written to SQLite (`telemetry.db`) as it
happens: agent, tools, guards, latency, tokens, message and reply. It survives
restarts, which the in-memory counters do not.

```bash
curl -s 'localhost:8000/metrics/history?hours=168&bucket=day' | python -m json.tool
```

`/metrics/dashboard` draws it. Four health indicators over time with their own
warn/problem thresholds, plus cost, speed and where the work went. A rise in
**guards fired** is the clearest sign the model is degrading rather than the
app. Thresholds live in `telemetry.py` and are served to the page, so the API
and the dashboard cannot disagree about what amber means.

The log holds full transcripts including any digits typed during verification,
so it is gitignored. `CHATBOT_TELEMETRY=off` disables it.

**LangSmith.** Optional, off unless configured. Each turn appears as a run tree
with triage, the specialist and every tool nested beneath it.

```bash
pip install langsmith
export LANGSMITH_TRACING=true LANGSMITH_API_KEY=lsv2_...
```

**The scoreboard.** `eval_metrics.py` defines the metrics, scores a run and
writes it to `eval_runs/` so the next run can be compared against it. Retrieval
needs no model, so it gates every push in CI:

```bash
python eval_report.py                      # score retrieval now
python eval_report.py --compare-only       # last two runs
python eval_report.py --fail-on-regression # exit 1 if anything dropped
```

---

## Testing

```bash
pytest                                                          # 346 tests
python simulate_customer.py --url http://localhost:8001          # needs a model
python simulate_customer.py --url http://localhost:8001 --scorecard
```

The simulator plays scripted customers against the running HTTP stack and
checks two different kinds of thing. Facts get assertions: did a tool actually
run, did verification hold, did another customer's order appear, are the
buttons resolvable. Judgements get an LLM judge: did it answer the question,
did it ask for what it needed, did it stay on topic. Matching on strings is a
bad way to test a judgement, because `contains("customer id")` fails a correct
answer phrased as "which account is this?" and passes a wrong one that happens
to contain the words.

Unit tests passing is not evidence the app works. That was proved here twice,
once by a mutation test and once when `"tools": []` turned out to be a 400 from
real vLLM and an accepted request in the mock, so 160 tests passed while every
turn failed. A fixture more permissive than the real thing is worse than no
fixture.

CI runs the suite on Python 3.10 through 3.12, scores retrieval, drives the
full HTTP stack against the mock server, asserts the verification gate refuses
before and allows after, and builds and boots the Docker image.

---

## Configuration

Precedence: a value already in the environment wins over `.env`, which is read
at startup by `env_file.py`. Environment variables do not cross terminal
windows; put them in `.env` and every process agrees.

```bash
cp .env.example .env
```

| Variable | Default | Effect |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Enables Claude |
| `CHATBOT_BASE_URL` | unset | Your model server, e.g. `http://localhost:8000/v1` |
| `CHATBOT_MODEL` | provider default | Model id |
| `CHATBOT_PROVIDER` | `auto` | `auto` / `anthropic` / `openai` |
| `CHATBOT_API_KEY` | unset | Key for the chosen provider |
| `CHATBOT_ENGINE` | `auto` | `off` disables the model, so every turn answers "unavailable" |
| `CHATBOT_MAX_TOKENS` | `1024` | Reply length cap |
| `CHATBOT_MAX_STEPS` | `6` | Tool rounds per message |
| `CHATBOT_FORCE_POLICY_SEARCH` | unset | `1` makes the policy agent's first call a tool call rather than asking it to search in the prompt |
| `CHATBOT_THINKING` | unset | `1` keeps reasoning models' chain of thought |
| `CHATBOT_TIMEOUT` | `120` | Seconds before a model request is abandoned |
| `CHATBOT_DISABLE_EMBEDDINGS` | unset | Force lexical-only retrieval |
| `RATE_LIMIT_PER_MINUTE` | `20` | Requests per minute per session or IP |
| `CHATBOT_TELEMETRY` | `on` | `off` stops recording turns to disk |
| `CHATBOT_CUSTOMER_DB` | `customer.db` | Read customers from SQLite when the file exists |
| `LANGSMITH_TRACING` | unset | `true` sends per-run traces to LangSmith |
| `API_PORT` | `8000` | Port the app listens on |

`.env.example` lists the rest with notes.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/ui` | The chat interface |
| `GET` | `/health` | Liveness |
| `GET` | `/status` | Engine, retrieval backend, record counts |
| `GET` | `/metrics/detail` | Percentiles, chat turns, tokens, cache hit rate |
| `GET` | `/metrics/history` | Model health over time (`?hours=168&bucket=day`) |
| `GET` | `/metrics/turns` | The last N turns as recorded (`?limit=50`) |
| `GET` | `/metrics/dashboard` | All of it, as a page |
| `POST` | `/api/chat/session` | Start a conversation |
| `POST` | `/api/chat/message` | Send a message, get a reply |
| `GET` | `/api/chat/history/{id}` | Full transcript |
| `POST` | `/api/search-knowledge` | Search policies and FAQs directly |
| `GET` | `/docs` | OpenAPI explorer |

```bash
SID=$(curl -s -X POST localhost:8000/api/chat/session \
      -H 'Content-Type: application/json' -d '{}' \
      | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

curl -s -X POST localhost:8000/api/chat/message \
     -H 'Content-Type: application/json' \
     -d "{\"session_id\":\"$SID\",\"message\":\"can I send this back?\"}"
```

Sessions live in server memory. A restart invalidates them; the UI detects this
and starts a new session rather than failing.

---

## Deployment

```bash
docker compose up --build
ANTHROPIC_API_KEY=sk-ant-... docker compose up --build
```

Or directly:

```bash
docker build -t customerchatbot .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... customerchatbot
```

The image runs `--workers 1` because sessions are in-memory (see below).

---

## Project layout

```
app.py                    FastAPI server, routes, middleware
conversation_manager.py   Sessions, history, context
llm_agent.py              Triage, data and policy agents
llm_providers.py          Anthropic and OpenAI-compatible backends
mcp_server.py             Tool definitions (the only data access)
database.py               Record access: SQLite if present, generated if not
customer_db.py            SQLite customer store
seed_db.py                Build customer.db from the seeded records
retrieval.py              Chunking, BM25, concept expansion, RRF
security.py               Verification, rate limiting, input hygiene
response_cache.py         TTL cache for knowledge lookups only
telemetry.py              Persistent turn log and health rollups
tracing.py                Optional LangSmith tracing; a no-op when unset
eval_metrics.py           Metric registry, scoring, run history
eval_report.py            Scoreboard CLI
production_constants.py   Timeouts, limits and thresholds in one place
env_file.py               Reads .env at startup, no dependency

simulate_customer.py      End-to-end customer simulator
judge.py                  LLM judge for the simulator's judgement calls
trace_turn.py             Print every prompt, call and result in one turn
mock_vllm_server.py       Scripted stand-in for vLLM, used by tests
start_vllm.sh             Launch vLLM with tool calling enabled
local_model_setup.sh      Guided local-model setup for Apple Silicon

static/
  chat_interface.html     Chat UI, single self-contained file
  metrics_dashboard.html  Served at /metrics/dashboard

tests/
  test_retrieval.py       Retrieval benchmark, agent loop, triage, guards
  test_security.py        Verification gate, data isolation, caching, limits
  test_telemetry.py       Turn log persistence and health arithmetic
  test_customer_db.py     SQLite store, record equivalence, SQL injection
  test_eval_metrics.py    Scoring maths and regression detection
  test_regression.py      Bugs that must not come back
  test_chat.py            Conversation and session behaviour
  test_production.py      Tools and the data layer
pytest.ini                Test discovery and sys.path
```

Tests import the application modules by plain name (`import llm_agent`), which
works from `tests/` because `pytest.ini` puts the repo root on `pythonpath`.
There is no `conftest.py` and no installed package.

The dataset is generated from a fixed seed, so `CUST-10001` is the same person
on every restart and in every process. The global RNG state is saved and
restored around generation, so seeding the data does not make the rest of the
application's randomness predictable.

---

## Known limits

In rough order of how likely they are to matter.

- **Sessions are in-memory.** A restart clears every conversation, and more
  than one worker would split sessions across processes. Persisting
  `ConversationManager` to Redis or SQLite is the fix.
- **The records are read-only.** Nothing creates, cancels or modifies an order,
  so a conversation cannot change state. Adding writes means new tools behind
  the same gate.
- **Dates in a seeded database are frozen.** The generator computes them
  relative to `now`, so orders age past their return windows as real time
  passes. Re-run `seed_db.py --force` to reset them.
- **Phone confirmation is not authentication.** It stops a conversation reading
  an account it cannot identify, but anyone with a customer ID and the last
  digits of the phone number gets in, and there is no lockout across sessions.
  A real deployment needs a signed-in user in front of `/api/chat/*`.
- **No fallback.** When the model is down the bot says so and stops. That is
  deliberate, but it means uptime is the model's uptime.
- **Two model calls per turn.** Triage, then a specialist. On a laptop at
  ~13 tok/s that is roughly 3s + 15s.
- **Rate limiting is per-process and in-memory.** It caps requests per minute
  per session and per IP, which stops one runaway client. It is not a spend
  ceiling and it resets on restart.
- **The live API path is not exercised in CI.** CI uses a fake key to verify
  engine selection only; the tool loop is tested against a scripted client.
