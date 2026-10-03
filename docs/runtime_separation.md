# Runtime vs. code: what belongs to Git and what belongs to Astra

The rule: **code changes through Git; Astra's accumulated state belongs to the
runtime environment.** A running process must not have its live state sitting in
the same tree an editor and an AI coder are freely modifying, and a `git pull`
must never fight the running process over the same files.

## Classification

| Class | Examples | Tracked by Git? | Where it lives |
|---|---|---|---|
| Source code | `main.py`, `astra/*.py`, shims (`memory_store.py`, ...), `deploy.py` | **yes** | repo checkout |
| Configuration | `config/*.yaml`, `.gitignore`, `.gitattributes`, `main.bat` | **yes** | repo checkout |
| Schemas / constants | module constants in `astra/memory.py`, `astra/reading.py` | **yes** | repo checkout |
| Tests / fixtures | `tests/**`, `test_storage/**` | **yes** | repo checkout |
| Book fixtures | `storage/library/**` (source books + cached text) | **yes** | repo checkout (seeded to runtime) |
| **Persistent runtime state** | memories (`roum_model.json`, `self_model.json`, `relationship_model.json`, `ai_journal.json`, `history_log.json`), `presence.json`, `reading_state.json` | **no** | `runtime/storage/` |
| Generated state | command history, `runtime_state.json`, `admin_snapshots.json`, library position | **no** | `runtime/storage/`, `runtime/snapshots/` |
| Logs | `supervisor.log`, runtime failures | **no** | `runtime/logs/` |
| Caches | `__pycache__/`, `.pytest_cache/` | **no** | ignored in place |
| Snapshots | bounded live-screen snapshot history | **no** | `runtime/snapshots/` |
| Temporary files | `*.tmp` (atomic-write scratch) | **no** | alongside the file, ignored |
| Deployment artifacts | running version/commit record, control requests, locks | **no** | `runtime/deploy/`, `runtime/control/` |

`storage/library/books/` is a *fixture* (the user's raw books), not accumulated
state; `storage/library/<work>.txt` is the extracted text cached on ingestion.
Both are version-controlled source material and are seeded into the runtime, but
never deleted.

## How it works

- `astra/paths.py` resolves the runtime root: `$ASTRA_RUNTIME_DIR` if set,
  otherwise `<repo>/runtime`. Storage, logs, snapshots, deploy records and
  control files all live under it.
- `build_session()` uses the runtime storage by default. On first run it *seeds*
  from the legacy in-repo `storage/` by copying files that are not already
  present. The originals are left untouched, so nothing is lost and the change is
  reversible by removing `runtime/`. Passing an explicit `storage_dir` (as tests
  do) uses that path verbatim and never seeds.
- `.gitignore` keeps `runtime/` out of Git entirely. The legacy `storage/*.json`
  are ignored too (they are a seed source only), while `storage/library/**`
  remains tracked.

## The lifecycle

    Astra running
      -> new code developed / pulled in the checkout
      -> tests run (deploy refuses on failure)
      -> deploy requested
      -> supervisor writes runtime/control/restart.request
      -> runtime flushes state (ChatSession.close -> store flush -> drain) and exits
      -> supervisor starts the new code
      -> runtime loads the same runtime/storage
      -> Discord reconnects
      -> Astra continues

Process continuity is **not** required. State continuity **is**. The supervisor
owns no state; it only starts, watches, backs off on crashes, and asks the
runtime to stop gracefully.

## Admin console

The CLI is a backend/administrative console now; Discord is the conversational
client. The console can be closed and reopened without disturbing the runtime,
because it connects to the same running components through the shared
application router rather than owning them. Type `menu` (or `/menu`) for the
numbered admin menu, and `/status` for the runtime version block.
