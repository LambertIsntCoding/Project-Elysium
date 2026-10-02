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
| `astra/affect.py` | Astra's current *experiential* affect (temporary, derived from experiences) |
| `astra/selfhood.py` | self-knowledge: the non-human boundary, the epistemic stance, absent-experience guard, derived self-portrait, formative/traumatic experience classification (pure, no I/O) |
| `astra/temporal.py` | Astra's sense of elapsed time: session gaps, long-open questions, long projects (pure render, never stored) |
| `astra/orchestrator.py` | prompt assembly and retrieval |
| `astra/consolidator.py` | turn -> governed memory decisions |
| `astra/inquiry.py` | questions, uncertainty, and revisable work knowledge (Slice 2) |
| `astra/reading.py` | reading vocabulary + the reader's idle/game/load gate (pure, no I/O) |
| `astra/library.py` | local ingestion (text/epub) + resumable reading position |
| `astra/reader.py` | the low-priority background reader daemon |
| `astra/elysium.py` | application-level root command layer |
| `main.py` | CLI, routing, slash commands |
| `config/*.yaml` | identity, relationship boundaries, style examples |
| `storage/*.json` | persistent memory (never hand-edit; never migrate blindly) |

## Maintenance scripts

| Script | Role |
|---|---|
| `curate_self_model.py` | One-off: retires stored self-records that contradict the boundary or are operational chatter, and demotes absent-experience claims. Nothing is deleted; idempotent (`--report` to preview). |
| `repair_null_fields.py` | One-off: strips no-op `null` placeholder fields from the store. |

Top-level `memory_store.py`, `orchestrator.py`, `elysium.py`, `consolidator.py`,
`memory_authority.py`, `affect.py`, `inquiry.py`, `reading.py`, `library.py`,
`selfhood.py` and `reader.py` are compatibility shims re-exporting `astra.*`. Import from
`astra.*` in new code.

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
- **Self-knowledge is bounded, and Astra's history is only her own.** The
  non-human boundary (`selfhood.NONHUMAN_BOUNDARY`) is settled knowledge, always
  in the prompt, never a memory and never an open question; a self-claim that she
  can become biologically human is kept only as a low-confidence
  `boundary_violation` observation and never shown as self-knowledge. A
  first-person claim about an experience she could not have had - a body, a
  childhood, a physical place, or a restatement of Roum's own life - is demoted
  in `add_memory` to a weak `absent_experience` observation (`selfhood.reifies_absent_experience`,
  `mirrors_roum_experience`). Quoted passages are stripped first, so a book's
  narration is never mistaken for her memory.
- **Healthy doubt is a stance, not a memory.** `selfhood.EPISTEMIC_STANCE` is
  injected every turn beside the boundary: she may ask why she thinks something
  is true and need not accept a claim merely because Roum or a source said it,
  but doubt must not collapse into refusal. It is a rendering, never stored, so
  it cannot decay, be reinforced, or be quoted back as one of her beliefs.
- **A self-belief is gated like a self-preference.** `self_belief` is in
  `SELF_DURABLE_CLASSIFICATIONS`: one generated sentence is stored as a
  `self_observation` (confidence capped at `SELF_OBSERVATION_CONFIDENCE_CEILING`)
  and only repeated evidence promotes it. Do not relax this to let the model
  narrate an identity into existence.
- **The self-portrait is derived, never stored.** `selfhood.derive_dispositions`
  reads Astra's own `experience` records and returns patterns with their
  evidence; the orchestrator renders it as an explicitly fallible block. It is
  never a memory type, so nothing about her can become a durable trait just
  because the model said it once.
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
- **Astra never sees her own numbers.** `affect.prompt_block`/`render_summary` and
  `relational.prompt_block` describe her state in plain language; the internal
  values that produce it are implementation, not introspection (implementation ->
  internal state -> experience -> self-interpretation). Do not reintroduce
  numeric readouts into the prompt.
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
- **A claim to an experience she never had is history, not knowledge.** A
  self-record that reifies an absent experience (Roum's life, a body, a
  childhood) is demoted at write time (`absent_experience`), and the orchestrator
  drops `absent_experience`/`boundary_violation` records before the experience
  split so they reach *no* prompt block. Kept on disk for audit; never surfaced
  as one of her recollections. This is what stops her making a personal
  experience out of something she did not live.
- **Time is felt, not prescribed.** `astra/temporal.py` renders the gap since
  `store.mark_present()` was last called, long-open questions, and long works.
  It is descriptive only - a gap is reported as elapsed time, never as a feeling
  the model must have. `mark_present()` is called on a real turn / explicit
  command only; `build_prompt` must never call it, or assembling the prompt would
  erase the very gap it is reporting (asserted in `tests/test_temporal.py`).
  `presence.json` is a fact about time, not a memory: keep it out of the stores.
- **`/library-scan` is read-only; the number it prints is the pick.** It lists
  the books folder (default `storage/library/books`, overridable) and never
  ingests. `/read add <n>` selects by that index, so a filename with spaces is
  never typed. A full path must be quoted; `split_args()` in `main.py` keeps
  quotes whole (`user_input.split()` did not, which is what broke spaced paths).
  A title override after the path renames the work in place via `save_state`,
  so it never spawns a duplicate.
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
- **Experiences are a primitive, and experiential affect is separate from
  relational state.** An experience is a self-model memory of type `experience`
  (kind, `work_id`, intensity, significance), written via
  `TripleMemoryStore.record_experience`. It reuses the ordinary evidence/decay/
  dormancy/supersession machinery, so isolated events fade and repeated ones
  persist - but it is never a `self_fact`/`self_preference`, so one experience
  can never redefine Astra's personality (that stays gated on repeated
  evidence). Recording an experience also nudges the *temporary* affect state in
  `astra/affect.py`, persisted in one self record tagged `experiential_affect`
  and kept out of generic retrieval. Affect components decay toward neutral
  (0.0), are surfaced by their own prompt block only when non-neutral, and
  change *processing* (retrieval breadth) rather than wording. `affect.py` and
  `relational.py` share accumulator mechanics on purpose but are deliberately
  NOT merged: a book can absorb Astra without changing how she feels about Roum,
  and vice versa. Experiences and affect reach the prompt only as Astra's own
  history/state, never mislabelled as facts about Roum or tentative inferences.
  Adding an affect dimension without a concrete behavioural consumer is
  discouraged; uncertainty belongs to the question system, not a scalar here.
- **Questions and uncertainty are memories, not a parallel store.** Slice 2 adds
  `astra/inquiry.py`, which is a *vocabulary* module only - the records are
  ordinary memories written through `add_memory`, so they inherit evidence,
  confidence, decay, dormancy, contradiction handling and supersession. An
  unresolved line of inquiry is a self-model memory of type `open_question`; a
  question's lifecycle (open/answered/reopened/abandoned) travels as a
  `qstatus:` *tag*, NOT as the memory status, so answering a question never
  deletes it - the history stays readable. "Known", "tentative" and "unresolved"
  are kept distinct by an `epistemic:` tag, and `CONFIDENCE_CEILING` caps how
  certain a record may claim to be (an `interpretation` can never look like a
  fact, however confidently the model worded it). Work-specific understanding
  (`observation` / `interpretation` / `hypothesis`) is filed under `roum` but
  scoped by `work_id`, so two works that share a name never blend; the
  orchestrator presents it in its own labelled block and keeps it out of generic
  retrieval. Revision rules are unchanged and non-negotiable: only an explicit
  correction may supersede, an inference weakens, and a conflict the evidence
  does not settle is *recorded* by `link_conflict` (which weakens neither side)
  rather than silently decided. `associate_evidence` links evidence by contextual
  overlap and never resolves a question on its own. Reasons to speak are exposed
  as `conversation_candidates` and are preserved only - nothing schedules or
  sends them. Detectors stay narrow: a false question or a false conflict
  corrupts the epistemic state, so prefer missing one over inventing it. Note
  `_relevant_work_knowledge` and `_select_relevant_questions` gate on
  `inquiry.significant_tokens` (stopword-filtered), because the shared retriever
  does not filter stopwords and a bare "the" would otherwise pull in another
  work's context.
- **Reading is a background courtesy, not a personality, and it writes no
  knowledge of its own.** `astra/reading.py` is pure vocabulary + policy (no
  store, no model, no thread); `astra/library.py` owns the local text and the
  reading *position*; `astra/reader.py` is the only thread. The reader wakes only
  when `reading.should_read` says so - Roum idle, no game, machine not loaded -
  and a game (or an explicitly busy GPU) always wins, because the GPU is shared.
  A cycle is bounded and re-checks the gate *between* chunks, so interaction or a
  game stops it immediately. Progress is saved after every chunk, and a **failed
  model call does not advance the offset** (`_extract` returns `None`, distinct
  from `{}`), so an interruption loses at most the chunk in flight and never
  skips a passage. The reader makes **one model call per bounded chunk** (never
  one per sentence) and reuses `orchestrator.query_gemma`; parsing is stdlib-only
  (text + EPUB) and fully offline. What reading *produces* is ordinary, governed
  memories via `apply_reading_results` (observations / interpretations /
  hypotheses / open questions, scoped by `work_id`) - never a parallel belief
  store, and never a durable `self_fact`/`self_preference` from a single
  experience. An interpretation is filed as `ai_inference`, so it can never
  supersede an explicit memory; a passage's unresolved question stays an open
  question and is not answered from general knowledge. The **reading position is
  deliberately not a memory** (it would rewrite the self model every chunk and
  break the byte-for-byte read guarantee); it lives in the library state file,
  which is the single source of truth for resuming. The orchestrator only *reads*
  the position for its own prompt block - prompt building stays read-only. Game
  detection uses `psutil` **only if installed**; without it the reader never
  invents a pause, and pausing relies on the session signalling one. `build_session`
  starts the reader by default (`enable_reader=False` to disable); the session
  marks activity on every input and re-checks for games each loop iteration.
