# Project Elysium

Astra: a local conversational companion. `main.py` is the CLI; the runtime lives
in the `astra/` package. Memory is stored as JSON under `storage/`.

## Run

```bash
python main.py            # start the CLI
python -m tests           # full suite (unittest + the two script-style suites)
python -m unittest tests.test_memory_governance -v   # one module
```

The model backend is Ollama at `http://localhost:11434/api/generate`
(`gemma4:e4b`). Tests never require it: routing is exercised against a loopback
HTTP stub, and consolidation tests call the decision layer directly.

## Layout

| Path | Role |
|---|---|
| `astra/memory.py` | store + governance policy (classification, authority, decay, utility) |
| `astra/orchestrator.py` | prompt assembly and retrieval |
| `astra/consolidator.py` | turn -> governed memory decisions |
| `astra/elysium.py` | application-level root command layer |
| `main.py` | CLI, routing, slash commands |
| `config/*.yaml` | identity, relationship boundaries, style examples |
| `storage/*.json` | persistent memory (never hand-edit; never migrate blindly) |

Top-level `memory_store.py`, `orchestrator.py`, `elysium.py`, `consolidator.py`
and `memory_authority.py` are compatibility shims re-exporting `astra.*`. Import
from `astra.*` in new code.

## Invariants

- **Nothing is deleted.** Memories are superseded, weakened, dormant or archived;
  all remain readable on disk.
- **Only an explicit user statement or correction may supersede.** Inference
  cannot override an explicit memory.
- **Directives about Astra are self-memories, not facts about Roum.** A sentence
  that addresses Astra ("Astra should stop narrating her analysis") is routed to
  the `self` model as a `self_observation`. `_routing_target` enforces this both
  in the consolidator and in `add_memory`, so no write path can file an
  instruction about Astra as a `roum` fact.
- **A restatement collapses, a contradiction weakens.** `detect_restatement`
  supersedes the older wording non-destructively, and is gated on
  `detect_contradiction` returning `None`, so a genuine polarity reversal still
  takes the "weakened" path. It requires a shared `_FEEDBACK_TAGS` tag (a shared
  topic keyword only corroborates) plus real content overlap; keep the
  `len(shared) >= 3/4` guards, because shared framing ("designated subject must",
  "currently experiencing") is not a shared subject. Prefer missing a collapse
  over superseding two genuinely different traits.
- **Sourced and unsourced material are separated in the prompt.** Only
  `SOURCED_SOURCES` (`is_sourced`) appear under `FACTUAL CONTEXT`; everything else
  goes under `TENTATIVE INFERENCES (UNVERIFIED - NOT STATED BY ROUM)`.
- **Prompt building is read-only.** Retrieval is recorded as *use* by
  `CompanionOrchestrator.note_retrieval` on the live turn path, never inside
  `build_prompt`, because the persistence tests assert byte-for-byte stability.
- **`storage/*.json` is not modified by tests** (persistence tests assert
  byte-for-byte stability).
- **Elysium is not a personality.** It is an application-level command route;
  an invocation must never reach `build_prompt()` or `query_gemma()`.
- The model proposes; the application decides. Classification heuristics live in
  `astra/memory.py`, not in the prompt.

## Notes

- `config/behavior_examples.yaml` must stay valid YAML: a parse error is
  swallowed and silently drops all style examples.
- Heuristic thresholds (self-promotion at 3 reinforcements, dormancy at 45 days,
  `GOVERNING_ACTIVE_MIN_CONFIDENCE`) are module constants in `astra/memory.py`.
- `detect_contradiction` matches on **containment** overlap (`shared / min(len)`)
  and only compares polarity on shared `_POLARITY_TERMS` that have a shared
  subject beyond the stance word. Jaccard union-overlap hid restatements of the
  same fact (verbose model-written sentences rarely share half their union), so
  supersession silently never fired. Keep the metric and the stance-word guard:
  loosening them either misses restatements or reverses unrelated memories.
- `/reconcile` replays contradiction resolution oldest-first so a store written
  before the detector recognised restatements heals itself. It is explicit, not
  part of `maybe_maintain`, because `storage/*.json` is never migrated blindly.
