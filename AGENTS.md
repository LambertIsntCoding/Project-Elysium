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
