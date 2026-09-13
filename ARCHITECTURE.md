# Architecture

## The core constraint

**No agent reads the database.** Every fact that reaches a customer came back
from a tool call. A generative model decides *what to look up and how to
explain it*; it never decides what is true.

## The shape

```
Browser ──POST /api/chat/message──▶ app.py
                                      │
                            ConversationManager
                     (session · history · context)
                                      │
                             ResponseGenerator
                                      │
                          Triage agent  (model, no tools)
                                      │  "data order_tracking"
                    ┌─────────────────┼─────────────────┐
              Data agent        Policy agent       General agent
              5 tools           2 tools            no tools
                    └─────────────────┼─────────────────┘
                            Gate  ·  Cache  ·  MCP tools
                    ┌─────────────────┴─────────────────┐
             Record database                    Knowledge base
       55 customers · 250 orders           20 policies · 30 FAQs
                                            (124 indexed passages)
```

Three agents, one model. This used to be a two-engine design — a model on one
side and a 1,200-line keyword-and-if-ladder rules engine on the other. The
rules engine is gone. What is left is infrastructure and agents.

## Why the work is split three ways

A tool loop asks a model two separate things: *which facts do I need*, and
*how do I explain them*. Measured on an 8B model running locally, the second
is reliable and the first is not — it skipped the lookup on roughly a third of
turns and answered from memory instead.

Splitting the job makes the first question smaller. Triage chooses between
three specialists with no tools to worry about. A specialist chooses between
two or four tools with a prompt about one job. Neither ever faces "six tools
and a prompt covering everything", which is the version that failed.

## Triage

Input: the whole conversation, plus whatever the tools have established.
Output: two words — the specialist, and the subject.

    data order_tracking
    policy shipping
    general greeting

There is no keyword router behind it. A fallback router would only decide the
cases it already agreed with, and would silently take the ones it did not; the
decision would stop being the agent's. When the answer names no specialist,
triage is told so and asked again (three attempts). If it still cannot, the
turn goes to the general agent — which has no tools, so an unroutable turn can
only produce a warm "tell me more", never an invented fact — and a
`triage_gave_up` counter is incremented and shown on the dashboard. The
customer is looked after; the failure is still visible.

The subject is a vocabulary, not a classifier: triage picks from a fixed list
so the label on a turn is a word the rest of the system understands.

## Specialists

| Agent | Tools | Job |
|---|---|---|
| data | begin_verification, confirm_phone, list_my_orders, get_order_details, check_return_eligibility | this customer's orders and account |
| policy | search_knowledge, list_policy_topics | the rules, same for everyone |
| general | none | greetings, thanks, nonsense |

Each is offered *only* its own schemas. Reaching for another agent's tool
returns an error naming what it does have, and the attempt is not recorded as
an action taken — otherwise a policy answer would report itself as having read
the account database.

## The verification gate

No account tool runs until the customer has confirmed the phone number on the
file they are asking about. `begin_verification` takes an identifier and
returns a masked hint plus a server-side challenge token; `confirm_phone`
compares the last seven digits with `secrets.compare_digest` and, on a match,
stamps the session verified for an hour.

This is enforced in `_run_tool`, not in the prompt. A prompt rule is a
request; a gate cannot be talked out of anything — an ungated tool call comes
back as a refusal the agent has to work around, and the refusal is not counted
as an action taken.

`list_my_orders` **takes no arguments**. Identity comes from the verified
session, so there is no parameter for the model to fill with someone else's
customer ID. Tools that do take an ID (`get_order_details`,
`check_return_eligibility`) receive `owner_customer_id` from the session and
check ownership in the data layer. That is the isolation boundary: not a rule
the model follows, a signature it cannot misuse.

Non-phone input (`"Return an item"`) is rejected as `not_a_phone_number`
*without* burning one of the five attempts; only an actual guess counts.

## Caching

A five-minute TTL cache sits in front of knowledge lookups only —
`search_knowledge` and `list_policy_topics`. `assert_cacheable` raises on
anything else, so an account tool cannot be added to the cacheable set by
accident and start serving one customer's orders to another. Failures are
never cached.

## Guards

Five, aimed at a model *faking* work or going in circles rather than choosing
badly:

- **Template leak** — the reply contains `<tool_call>`, `[TOOL_CALLS]` or
  similar as plain text, meaning the model role-played the exchange and
  invented the result.
- **Narration** — "one moment, let me check" with no tool call behind it. The
  customer gets a promise and no answer.
- **Deflection** (policy only) — "you'd need to contact support about that"
  when no tool ran. The policy agent *is* support; it has the knowledge base.
- **Ungrounded fact claim** (policy only) — a specific number, window or term
  stated with `tools_used` empty, i.e. recited from training data.
- **Repeat reply** — the answer is, word for word, the one the customer just
  replied to. Compared on normalised text, and only above 40 characters, so
  "No problem." is still allowed to recur.

Each gets one correction. A second offence raises, and the turn is reported as
unavailable rather than shown to a customer — except the repeat guard, which
ships the reply instead: a model that repeats itself twice will not be argued
out of it, and looping would spend the step budget to send nothing at all.

### Where a tool call is not optional

Two steps send `tool_choice: "required"` instead of `"auto"`, because at both
of them every legitimate continuation is a tool call:

- **The policy agent's opening move** — search before answering. Off unless
  `CHATBOT_FORCE_POLICY_SEARCH=1`.
- **The step straight after the identity gate opens** — always on.

The second is the more interesting one. Verification is plumbing: the customer
asked to track an order, and that a phone number matched is something the
system needed, not something they wanted told. But `confirm_phone` returns
`{"verified": true}`, the model reads a tool result as an event worth
reporting, and the turn ends on "Great, you're confirmed — let me check your
orders" with nothing looked up.

That is not promptable. After a `tool` role message the cheapest continuation
for a small model is prose summarising it; that is what the chat template
trained it to do, and `"Saying you will act, without acting, is the worst
thing you can do"` does not outbid it. Removing the choice does.

It is safe here in a way it is not at step 0, where "what's your customer ID?"
is a correct reply and a forced call would break verification outright. Here
the gate is open, triage has already decided the turn needs account data, and
there is nothing left to ask. A *failed* `confirm_phone` does not trigger it —
a wrong number needs a sentence, not a lookup — and the flag clears as soon as
an account tool has run, so the model gets its choice back to write the answer.

Servers that reject `required` degrade to `"auto"` on the first rejection, with
a warning, rather than failing every verification turn from then on.

### Why the repeat guard exists

It is the only guard aimed at a model that is behaving *correctly*. Asked "who
are you?" twice in the same state, the model answers the same way twice, which
is the right response to the prompt it can see. The bug was upstream: the
context note described an unchanged world.

That is fixed at the source too — `identity_asks` and `failed_lookups` now
make an unverified turn read differently from the one before it, and the
system prompt is rebuilt after every tool round so a mid-turn verification is
visible to the next step. The guard stays because those fixes are prompt-shaped
and this one is not: it holds whatever the note says.

Every guard here started as a line in a prompt that the model obeyed until it
had a bad day. Prompts hold most of the time; guards hold always. Behaviour
that matters gets promoted from prose into code.

## Turn labelling

Two independent axes, deliberately separate fields:

- `intent` — the **subject**: shipping, payment, returns, order_status…
- `answer_source` — where the facts came from: `knowledge_base`,
  `account_data`, `none`

"How much is shipping" is a shipping question *and* a policy question, so
those were never alternatives. The subject comes from what was actually looked
up (the category of the retrieved passage, or the account tool that ran), and
falls back to triage's word when nothing was looked up.

## Conversation state

`ConversationContext` holds only what a tool returned (`customer_id`,
`current_order_id`, `known_order_ids`, `current_order_returnable`) plus the
identity state the gate needs (`verified_customer_id`, `verified_at`,
`challenge_token`, `verification_failed`). Everything else — what the customer
meant, what they already said, what they are waiting on — the agents read from
the transcript, which they have in full. A flag would only be a worse copy
of it.

Two counters are the exception: `identity_asks` and `failed_lookups`. The
transcript *does* contain both facts, but the context note is appended to the
end of the system prompt — the most salient position for a small model — and
it was rebuilding identically on every unverified turn. The model was reading
"ask who they are" as the last instruction in its prompt and doing exactly
that, twice. These two counters are what make the second turn's prompt
different from the first's. They store counts only: an unverified conversation
holds no identity, and a wrong identifier is still somebody's data.

## Retrieval

Passage-level chunking, BM25 with concept and phrase expansion, optionally
fused with dense embeddings via Reciprocal Rank Fusion (k=60). The policy
agent's `search_knowledge` is the only path to it.

Without `model2vec` installed it is lexical-only, and `/status` says so.

## When the model is unavailable

The customer is told plainly and the turn ends. There is no keyword engine
answering in its place, because a support bot that quietly answers from a
keyword table is wrong without telling you.

## Testing

| Layer | What it catches |
|---|---|
| `pytest` | units, wire formats, regressions, agent structure |
| `simulate_customer.py` | whole-stack behaviour over HTTP, full transcripts |
| `trace_turn.py` | every prompt, call and result in one turn |

Unit tests passing is not evidence the app works — that was proved here twice.
Once with a mutation test; once when `"tools": []` turned out to be a 400 from
vLLM and an accepted request in the mock, so 160 tests passed while every
single turn failed in production. **A fixture that is more permissive than the
real server is the one kind of fixture that actively lies to you.** The mock
now returns the same 400.

The simulator is the layer that catches "technically a 200 response, obviously
broken to a person". Its judgement calls go to an LLM judge (`judge.py`) rather
than to string matching — whether a reply is *evasive* or *reassuring* is not a
substring. Facts and security properties stay on deterministic assertions,
because those have one right answer and a judge would only add variance.
