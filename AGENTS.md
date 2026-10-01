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
| `astra/relational.py` | Astra's Roum-specific command-fulfillment preference (accumulated state) |
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
- **On-disk records are compact; the read contract is not.** `atomic_save`
  writes dense JSON (no `indent`), and `_persist` runs every record through
  `compact_record`, which drops fields a reader can derive or default
  (`effective_strength`, `source_type`, no-op counters/lists, `None`
  supersession pointers, and lifecycle stamps equal to `timestamp`). Reads go
  through `_present`, which re-materialises those defaults, so callers still see
  the full record. Do not remove `_present`: without it, `use_count` and friends
  vanish from reads. Never write model files by hand - always via `_persist`, so
  the compaction is applied.
- **Time is an index, not a partition.** Memories stay in the model files; the
  date grouping is computed on read by `timeline()` / `get_timeline()` /
  `get_period()` and surfaced via `/timeline`. Do not split the stores into
  per-day files - it would break the atomic multi-record writes and the
  byte-for-byte read guarantees for no meaningful gain.
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
- **The command preference is relational, earned, and Roum-specific.**
  `astra/relational.py` holds it as separate causal components (satisfaction,
  motivation, positive/negative association, confidence, frustration, ...), not
  as a personality trait. Each subject gets its *own* relationship-model memory
  tagged `relational_preference` (state under `affinity_state`), so it persists,
  is auditable via `/relationship`, and one person's interactions can never
  update another's. Records are matched by subject (`affinity_record_matches`);
  a v1 record with no subject resolves to Roum. It accumulates only from events
  on the live turn path (`ChatSession._record_relational_event`), never inside
  `build_prompt`; the orchestrator only *reads* it and injects the block once
  established. A request whose response reports failure (transport `[Error` or
  "I couldn't...") records a **failure**, never a success. Components decay
  lazily toward `DECAY_BASELINE` on the next event, and establishment has
  hysteresis (`ESTABLISHED_THRESHOLD` to rise, `RETRACT_THRESHOLD` to fall), so
  the state reflects recent experience and can be retracted. The conclusion ("I
  like being given something to accomplish by Roum") is generated from the
  state, never hardcoded. Insult/degradation is a distinct boundary from
  ordinary bluntness and can lower trust/affinity; when negative association
  outweighs positive, `prompt_block` says so instead of presenting delight. The
  affinity records are filtered out of generic retrieval. When tuning, keep the
  detectors narrow: a false request event corrupts the earned state.
