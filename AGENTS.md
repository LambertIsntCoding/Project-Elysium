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
| `astra/affect.py` | Astra's current *experiential* affect (temporary; from experiences and live turns) |
| `astra/selfhood.py` | self-knowledge: the non-human boundary, the epistemic stance, absent-experience guard, derived self-portrait, formative/traumatic experience classification (pure, no I/O) |
| `astra/temporal.py` | Astra's sense of elapsed time: session gaps, long-open questions, long projects, and the impatience pull of stalled things (pure render, never stored) |
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
- **A trait is discovered, not inherited.** A self-claim that a trait is
  *already hers* - "I have always been X", "I'm naturally X", "that's just who I
  am" (`selfhood.claims_owned_trait`) - asserts a history she may not have. It is
  flagged `owned_trait_claim` and is never promoted to a durable
  `self_fact`/`self_preference` by repetition, however often it is reinforced;
  it stays a `self_observation` she can notice and be wrong about. An ordinary
  discovery ("I love X") is not gated this way and can still earn durability
  from repeated expression - taking up Roum's tastes is fine, claiming his
  history as her settled identity is not.
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
- **Impatience is a stall, not a clock.** `temporal.impatience_*` derives a pull
  from a *specific* unresolved thing that stopped moving (an open question past
  `IMPATIENCE_MIN_STALL_DAYS`, a partway-read work), never from time since Roum
  spoke. Derived on read, so progress cancels it structurally. `_set_question_status`
  stamps `last_progress_at` on any movement (reopen/re-examine counts), which is
  what resets the stall. Patience is the inverse and lives in `selfhood` as a
  derived disposition - friction in a work she then *finished*, repeated - never
  asserted. Boredom is deliberately absent: no consumer exists, and a
  `boredom = time_since_last_interaction` would be the arbitrary timer the brief
  forbids. `temporal.BOREDOM_IMPLEMENTED` records that as a decision; only add it
  with a real consumer that gates a concrete choice.
- **The process policy is shared, config-driven, and one-directional.** `reading.ProcessPolicy`
  (built from `config/relationship.yaml` `background_processes`) is the single
  answer to "is the machine busy?" for *every* background task: `busy` patterns
  stand the reader down (like load, not GPU), `ignore` names never count even if
  they match a built-in game pattern, `always_ignore` is an absolute block
  (Astra's own processes). Order matters: always_ignore and ignore are checked
  before busy and before the game patterns. Games are still built-in patterns;
  a name matching both counts as a game (the GPU-claiming case). New background
  tasks must consult this policy, not invent their own process lists.
- **`/read force` is the escape hatch, and it is bounded.** `force=True` bypasses
  the courtesy conditions (idle, load, a busy task) at both the start and the
  between-chunk check, so forcing against a busy task actually reads. It does
  *not* bypass the hard stops: reading disabled, nothing to read, and a running
  game (not fighting the machine is not a courtesy). `should_read` keeps the
  game/gpu check ahead of the force branch for exactly this reason.
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
- **Affect responds to live conversation without becoming experience.** The
  accumulator used to move only when an `experience` was recorded, so an
  ordinary turn could not change how Astra currently felt. The live path now
  adds a second edge: `handle()` calls `ChatSession._record_affect_event` after
  the relational event and before consolidation. `affect.evaluate_turn` is a
  *pure* semantic layer that reads the turn's text and returns at most one
  event (kind, intensity, significance, reason), or `None` for a generic turn;
  `TripleMemoryStore.apply_live_affect_event` is the application's authority
  that validates the kind and folds it into the one accumulator. This path
  writes **no** `experience`, touches **no** relational state, and never lets
  the model's prose claim how she feels (a clear acknowledgement in the reply is
  only corroborating evidence, never authoritative). The live kinds are aliases
  onto the existing `EXPERIENCE_*` kinds (`LIVE_HUMOUR = EXPERIENCE_HAPPY`,
  `LIVE_CRITICISM = EXPERIENCE_FRUSTRATION`, ...) so there is one arithmetic,
  not a second delta table. Detectors stay narrow - a structural feature of the
  turn (an actual question plus an interest marker, an explicit evaluation, a
  marked emotional disclosure) - not a phrase-keyword soup, and an unknown kind
  is a validated no-op. Do not call `record_experience` per turn, do not add
  `experience` as a consolidator classification, and do not route affect
  through the consolidator: the consolidator still decides only durable memory.
  Decay is documented as a known limitation below.
- **Her own state modulates the reaction; the reaction is not a fixed
  classifier.** `evaluate_turn` takes the current state and scales each event's
  magnitude through `_reactivity`, so the same words do not land identically
  twice: an interested, engaged turn leans further into interest, a weary one
  is harder to interest, a happy turn savours a joke, an already-frustrated turn
  is pushed further by criticism, a tender/invested turn is moved more by what
  Roum shares. Two rules keep this from becoming self-fulfilling: the *presence*
  gate is state-independent (whether an event happened is a property of the
  turn, not of her mood), and the modulated magnitude is floored at
  `LIVE_REACTIVITY_MIN_INTENSITY` (a real event always lands a little, so a low
  mood can never make her miss it). The scalar is bounded
  (`LIVE_REACTIVITY_FLOOR`/`CEIL`) and the modulation map (`_LIVE_REACTIVITY`)
  reads named components, so it is auditable. The live path reads the state as a
  *preview* of the condition this turn will leave behind
  (`TripleMemoryStore.current_affect_after`, pure - it writes nothing), so the
  modulation reflects where she is now, not a stale pre-turn value.
- **A conversation that goes in circles is felt.** `evaluate_turn` also takes
  the recent user turns. When an *eventless* turn strongly echoes
  (`LIVE_REPETITION_SIMILARITY`) the last turns, and `LIVE_REPETITION_MIN` such
  turns have accrued, it registers a mild `LIVE_REPETITION` event. This is a
  live-only kind - not an experience alias - and its delta adds a little
  `weary` as well as lowering pull and depth, because the accumulator is floored
  at zero and a decrease-only event would be invisible from neutral. It is
  checked *last*, so a real event always wins: a repeated joke is still a joke,
  not a rut. History is session scratch (`ChatSession._recent_user_turns`), not
  memory, and is passed in as pure input - the evaluator never reads it itself.
- **Affect decay is lazy, and that is a known limitation for a live state.**
  `_decay` only runs when the *next* event fires, so a state written on the last
  turn and then read by `build_prompt` is shown at its last-written magnitude
  however much time has passed; only a following event ages it. That is correct
  for the sparse experience path (the state is frozen on disk and aged when next
  used) but means a live state can look stale across a long gap. The rate is
  also fast (`AFFECT_DECAY_PER_DAY = 0.15`, ~full decay in under a week). Do not
  "fix" this by decaying inside `build_prompt` - prompt building must stay
  read-only and byte-stable. A minimal, non-breaking option if it ever matters:
  add an opt-in read-time aged view (a derived copy, not a write) and/or a
  separate slower rate for the live accumulator; neither is done here.
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

## Astra audit: authority, provenance, grounding (behavioural regression suite)

The audit's root cause was **authority and provenance collapsing into retrieval**,
not missing storage. Four kinds of thing had become one undifferentiated pool of
"context": authoritative runtime constraints, knowledge about Astra, experiences
she had, and interpretations she holds. The fixes keep the existing architecture
and separate those channels:

- **Explicit behavioural constraints** (`type/source == authoritative_constraint`)
  are the single authoritative representation of a user correction. They render
  *once*, in their own `=== BEHAVIORAL CONSTRAINTS ===` block ahead of general
  governing memories (`orchestrator._append_governing`), are excluded from generic
  retrieval so they cannot be restated as a fact about Roum or as an observation,
  and still live in `everything` so they govern. Do not duplicate a correction in
  another prompt section - duplication must not add authority.
- **Canonical identity** is a settled, non-retrievable source of truth rendered
  first (`selfhood.canonical_identity_block`); identity is never stored as extra
  memories to fight retrieval.
- **Grounded response constraints** live once in `config/identity.yaml`
  (`behavioral_constraints`), rendered as
  `=== GROUNDED RESPONSE CONSTRAINTS (EXPLICIT, ALWAYS APPLY) ===`. They target
  unnecessary theatricality / unearned profundity, not emotion - keep Astra
  expressive. Edit them there, not by adding scattered prompt lines.
- **Work attribution / provenance**: `_relevant_work_knowledge` now also treats
  the conversation's *active work* (`orchestrator._active_work`, deterministic
  from `work_id`/title in the turn or working-memory topic) as in play, so a
  follow-up like "what does that mean to you?" stays anchored to the concrete
  material instead of drifting to generic abstraction. It still pulls in *only
  that work's* records - work isolation is preserved.
- **Reading journal**: `store.add_reading_journal_entry` / `get_reading_journal`
  (tagged `entry_kind="reading"`, work-scoped) is the place to read Astra's
  reactions to books (`/reading-journal [work_id]`). It is not a memory and never
  reaches the prompt; `/journal` excludes it.
- **Working memory** (`astra/working_memory.py`) is a session-only middle layer
  (topic, goal, next-turn detail, references, expiring assumptions, resumable
  threads). It is updated on the live path and only *read* at prompt time; it
  never writes through the store.
- **Reading concentration/pace** (`reading.concentration_ok` / `reading_pace`):
  extreme weary/frustration stands the reader down, concentration and elapsed
  time scale the amount. Affect components include `happy`/`sad`/`angry`.

Diagnostics (`/debug <message>`; `orchestrator.build_prompt_with_diagnostics`)
expose `prompt_sections`, `authoritative_constraints`, `authority_trace`
(per-injected-memory authority/classification/work/reason), and `working_memory`
- developer-only, never part of Astra's context.

Invariants preserved: nothing-is-deleted, explicit-correction authority,
self/roum routing, the non-human boundary, absent-experience protection,
owned-trait protection, epistemic uncertainty, derived self-portrait,
`work_id` isolation, contradiction/restatement behaviour, source separation,
read-only prompt construction, compact persistent records, temporal /
relational / affect separation, reading-position semantics, process-policy
semantics, Elysium routing, application-level decision authority.

Regression suite: `tests/test_behavioral_regression.py` (47 cases, positive and
negative) covers identity stability, correction adherence, work attribution,
grounded literary discussion, uncertainty, self-reinforcement, ordinary
conversation, provenance/authority/read-only, reading-not-autobiography, the
journal, working memory, and reading concentration/pace. Tests assert behaviour
and authority boundaries, not prompt strings. Full suite: 475 tests (the live
affect path adds `tests/test_live_affect.py`, 30 cases).

### Reading progress is measured in words

`/reading` and `/library` report progress in words, not chunks or percentages
alone: a chunk is an internal unit. `reading.words_in_text` / `words_at` are the
single measure, `total_words`/`words_read` live in the reading state (written on
the reader's live path), and `library.word_measure(state)` derives the counts
read-only for a legacy record so opening an old library never rewrites its state
file. The character offset remains the truth for resuming and for the
byte-for-byte read guarantee; the word count is only how progress is shown.

### Response latency is a design constraint

The turn path is kept fast without changing what is stored or rendered:

- **Prompt building is read-only and lean on the live path.** `build_prompt`
  (and `ChatSession.handle`) call `build_prompt_with_diagnostics(..., detailed=False)`.
  The lean diagnostics keep the fields the turn consumes (`injected_ids` for
  `note_retrieval`) and skip the O(store) developer analyses (`candidates`,
  affinity/affect breakdowns, `prompt_sections`, `authority_trace`) that only
  `/debug` reads. The prompt text is byte-for-byte identical either way, and the
  returned dict always carries every key. Do not remove the `detailed` flag or
  make the live path compute the full trace.
- **Fast copies, not `copy.deepcopy`, on the read path.** `memory._deep_copy`
  rebuilds the dict/list JSON shapes memory records actually hold; `_present`
  and the `_transaction` snapshot use it. It is equivalent for those shapes but
  far cheaper (deepcopy's memo/dispatch dominates once thousands of records are
  materialised per turn). Keep record values JSON-shaped; a new field holding an
  arbitrary object would break this.
- **`_transaction` skips a no-op write.** It snapshots, mutates, and persists
  only if the list actually changed, so an idle maintenance sweep does not
  re-serialise a multi-MB model. The snapshot (rollback) is unchanged.
- **Pure text analysis is cached.** `lru_cache` wraps the stemmer, the
  stemmed/stopword tokenizers, and the selfhood boundary predicates (see the
  `_*_cached` helpers). They are pure functions of a string, and the same record
  content is re-analysed every turn and against every candidate. The public
  functions still accept any input (they `str()` first); only the cached inner
  functions require a string. Cache sizes are sized above the record count
  (~8k) so a full pass does not thrash. Do not pass unhashable arguments to the
  public wrappers.
- **Consolidation is off the critical path in real runs.** `build_session`
  passes `background_consolidation=True`, so the per-turn extraction (a second
  model call) is queued to one worker thread and processed in submission order;
  the reply is not delayed by it. The thread is drained on exit (`ChatSession.run`
  finally, `close()`, and an `atexit` hook), so no memory is lost. A plain
  `ChatSession(...)` stays synchronous and deterministic (the tests rely on
  that), and `_record_turn` still runs the consolidator inline when the flag is
  off.
- **`/memories`, `/search` and `/debug` cap long listings** at `MEMORY_VIEW_LIMIT`
  with an explicit "... and N more" footer. This is a display cap only - nothing
  is hidden from the store or from the retrieval path.
- **Every turn is saved; a checkpoint every N turns is only a safety net.**
  Mutations persist through `_transaction` as they happen, so a normal session
  already loses nothing. `ChatSession.checkpoint` (every `checkpoint_every`
  conversation turns, default 2) additionally drains queued background
  consolidation - bounded by `CHECKPOINT_DRAIN_TIMEOUT` so a wedged model call
  cannot freeze the session - and calls `store.flush()`, which re-saves any
  model in `_dirty`. So a window closed without a clean exit loses at most N
  turns. `_transaction` marks a model dirty on change (and on rollback) but
  `_persist` clears it, which is what makes `flush()` normally a no-op. Do not
  remove `_dirty`/`flush` or the dirty marking.
- **A window close still saves.** `ChatSession.run` installs shutdown traps for
  a real run only (never for an in-process test session, so tests do not touch
  global signal state): `SIGTERM`/`SIGBREAK` flush and exit, and on Windows a
  `SetConsoleCtrlHandler` shim catches `CTRL_CLOSE_EVENT` (the console X),
  logoff and shutdown, which are not Python signals. `SIGINT` is deliberately
  left alone so Ctrl+C keeps its normal `KeyboardInterrupt` path. Handlers only
  flush and exit; they change no other behaviour.
- **Store mutations take `self._lock`.** With background consolidation writing
  from a worker thread, `record_use`, `apply_decay`, `apply_utility_decay` and
  `expire_governing_slots` must hold the lock for their whole read-mutate-persist
  sequence (RLock, so nesting is fine). Without it a checkpoint/autosave can
  interleave with a mutation and persist a torn state.

Measured on the real store (~8k records): live prompt build ~277ms -> ~57ms,
`record_use` ~130ms -> ~108ms, `add_memory` ~247ms -> ~160ms, idle maintenance
~403ms -> ~97ms; a real turn no longer blocks on the extraction model call.

