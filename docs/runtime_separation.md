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

### Watching and driving a running runtime

Because a running Astra holds the single-instance lock, opening a second normal
console against the same runtime storage cannot read stdin - you would only be
able to send commands through Discord. The inspection console is the supported
way to watch *and* drive her from a separate process:

```bash
python main.py --inspect
```

It opens the same runtime state a real run uses and shows whether the runtime is
running (from `runtime/runtime.lock`), her autonomy/schedule state, Discord
connection health, background tasks, affect and relationship views, unresolved
questions, reading position, and the time sense. How it behaves depends on
whether a runtime is running:

- **A runtime is running.** Every command - including mutating ones such as
  `/sleep`, `/autonomy pause` or `/update apply` - is forwarded over a small
  filesystem *control channel* (`astra/control.py`) and executed by the running
  runtime through its own `ApplicationRouter`. The console is a genuine remote
  control, so the whole command surface is available and usable, and behaviour
  cannot drift from the CLI or Discord. Conversation is still never routed: only
  commands are forwarded, and a command issued this way never records a human
  message or presence side effect.
- **No runtime is running.** Nothing can be acted on, so it falls back to honest
  read-only views: the same `ChatSession` display methods and `RuntimeController`
  reports the live runtime uses, re-read from the durable stores before each
  command. A command that would change state is *refused* with a pointer to the
  runtime rather than silently doing nothing. Reads that only *look* like they
  might change something are still served: `/library-scan` lists the books
  folder and never ingests (so it is rendered, not refused), and `/update`
  performs a Git *fetch* to answer "is there an update?" - a fetch updates only
  remote-tracking refs, never the working tree - while `/update apply`/`reset`
  remain refused because they rewrite the checkout.

In both modes the console never calls the model directly and never opens
Discord; on a populated runtime it writes nothing (only a first run performs the
one-time storage seed). Type `/help` inside it for the command list.

The control channel is deliberately a filesystem mailbox, not a socket: the
runtime already coordinates through `runtime/control/`, so this adds no
listening port. A request carries only the command text and a source tag; a
result carries only the text the runtime produced. Both are written atomically
and consumed once, so a restart mid-request leaves at most one stale file, which
the console reports honestly rather than hanging. The runtime side runs in the
existing control-watcher thread (`ChatSession._install_control_watcher`).

### Updating the code with minimal downtime

`astra/update.py` is the "pull command": it brings the checkout up to date with
its remote and promotes the new code to the live runtime, reusing the existing
graceful-restart machinery rather than inventing a second one. It is reachable
four ways, all sharing one implementation:

```bash
python main.py --update              # check + pull (fast-forward) + restart
python main.py --update --update-mode reset   # hard-reset to the remote
```

- `/update` (CLI session, Discord, admin menu, and the `--inspect` console)
  reports whether the checkout is behind its remote;
- `/update apply` pulls (fast-forward) and restarts;
- `/update reset` hard-resets the checkout to the remote, then restarts.

Two guarantees, unchanged from `deploy.py`:

- **Runtime state is untouched.** Updating never copies, moves or edits the
  runtime tree; the running process is only asked to restart through its own
  flush path (`ChatSession.close` -> store flush -> drain consolidation), so the
  downtime window is that flush, typically well under a second.
- **Nothing is destructive.** The working tree must be clean (tracked files)
  before anything is fetched; a failed fetch/pull leaves the checkout exactly as
  it was, and a diverged branch is refused under `ff-only` rather than merged.

When the command is issued *by the running runtime itself* (`/update apply` on
the CLI or Discord), the runtime does not block waiting for its own restart - it
answers first, then the control watcher performs the graceful restart on its next
tick. When issued from a separate process (`--update`, or `--inspect` forwarding
to the runtime), the caller can wait for and verify the fresh start.

### Live screens refresh again

The live observation screens (`menu` -> Relationships & Affect -> Relationship /
Affect, and Diagnostics) redraw on their own cadence. This depends on
`ChatSession._screen_poll` returning `None` on a timeout: the earlier version
blocked on `input()` and so never refreshed on a real terminal. If a new screen
is added, keep using this poll and pass the session's `_screen_clear` as the
`clear` callback so frames replace themselves instead of scrolling.
