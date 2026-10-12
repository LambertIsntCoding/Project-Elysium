# Autonomous operation, scheduling, and proactive Discord

This document is the operator's guide to running Astra as a persistent,
headless service: what it does on its own, how to configure it, and how to
start, stop, inspect and verify it. It assumes the runtime/code separation in
`docs/runtime_separation.md`.

Nothing in this document widens a safety boundary. Schedules and limits are
*policy*; the hard guarantees (no unapproved identity/code/credential changes,
no invented facts, bounded resources) are enforced in code.

## 1. What runs

The runtime is a single process with several cooperating, bounded layers:

| Layer | Module | Role |
|---|---|---|
| Scheduler | `astra/scheduler.py` | Runs durable background tasks with resource/sleep policy. |
| Time arithmetic | `astra/schedules.py` | Pure: recurrence, windows, quiet periods, deadlines, expiry, DST-correct zones, catch-up policy. |
| Schedule driver | `astra/schedule_runtime.py` | Materialises configured schedules into durable tasks; keeps due instants current; misses/expires honestly; re-arms after a run. |
| Task store + ledger | `astra/tasks.py` | Durable queue (lifecycle states) and a bounded attempt ledger. |
| Activity planner | `astra/activity.py` | Bounded, evidence-backed selection of what to do next; can also decide idle. |
| Notification policy | `astra/notifications.py` | Decides *deliver now / digest later / drop*, and enforces limits, quiet hours and dedupe. |
| Autonomy driver | `astra/autonomy.py` | One bounded wake: advance schedules, run scheduled work, plan, consider outcomes, deliver a digest. |
| Discord adapter | `astra/discord_adapter.py` | Transport + delivery bookkeeping only. Inbound routing and outbound proactive send. |
| Reply policy | `astra/reply_policy.py` | Whether a reply should quote/ping a specific message. |

The **autonomy driver wakes on a bounded cadence** (`autonomy.wake_interval_seconds`,
default 30s). Nothing here polls at high frequency or busy-loops; a wake is a
cheap read that asks "is anything due?".

## 2. Run it

### Interactive console (default)

```bash
python main.py
```

### Persistent headless runtime

```bash
python main.py --headless        # no console loop; runs until stopped
python main.py --supervise       # under the supervisor (restart-on-crash, graceful redeploy)
python main.py --status          # print running version/commit
python main.py --restart         # ask the supervisor to restart gracefully
python main.py --stop            # ask the supervisor to stop gracefully
```

`--headless` keeps the process alive with no interactive console, runs the
Discord transport and the autonomy driver, and flushes state periodically. The
control watcher (`runtime/control/{restart,stop}.request`) lets a stop/restart
happen even when the console is blocked, flushing through the runtime's own
`close()` path.

Closing a CLI or conversation window does **not** stop the headless/supervised
runtime. To stop it, use `python main.py --stop` or stop the scheduled task.

### Secrets

The Discord token is read from an environment variable named in config
(`discord.bot_token_env`, default `DISCORD_BOT_TOKEN`). It is never written to
config or logs. Set it in the environment that launches the runtime.

## 3. Start automatically on Windows (Task Scheduler)

Run Astra headless at logon and keep it running. Use the "At log on" trigger and
a restart-on-failure policy.

1. Create `run_astra.cmd` (adjust paths):

   ```bat
   @echo off
   set DISCORD_BOT_TOKEN=your-token-here
   cd /d C:\path\to\Project-Elysium
   python main.py --supervise >> runtime\logs\astra.out 2>&1
   ```

   Store the token in a user environment variable instead of the script if you
   prefer; the runtime only reads it from the environment.

2. Register the task (PowerShell, run once):

   ```powershell
   $action  = New-ScheduledTaskAction -Execute "C:\path\to\run_astra.cmd"
   $trigger = New-ScheduledTaskTrigger -AtLogOn
   $settings = New-ScheduledTaskSettingsSet -RestartCount 999 `
       -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries
   Register-ScheduledTask -TaskName "AstraRuntime" -Action $action `
       -Trigger $trigger -Settings $settings -RunLevel Highest
   ```

3. Start it now: `Start-ScheduledTask -TaskName "AstraRuntime"`.
4. Inspect: `Get-ScheduledTaskInfo -TaskName "AstraRuntime"` and read
   `runtime\logs\astra.out`.

The supervisor holds `runtime/runtime.lock` (single-instance), so a duplicate
launch exits rather than running scheduled work twice or double-sending messages.

## 4. Configure schedules

Schedules live in `config/runtime.yaml` under `autonomy.schedules`. Each entry
names a task `kind` and a `dedupe_key` (its logical identity). See the header of
`astra/schedules.py` for the full shape.

```yaml
autonomy:
  schedules:
    - mode: daily            # daily | interval | once
      at: "03:30"            # daily: local HH:MM
      timezone: "America/New_York"   # IANA name (DST-correct)
      kind: self_review
      dedupe_key: "nightly-self-review"
      priority: 42
      start: "02:00"         # window: no earlier than this local time
      end: "05:00"           # window: no later than this local time
      deadline: "05:30"      # past this, a late run is recorded as missed
      catch_up: skip         # run | skip | coalesce
      expires: "2026-12-31T00:00:00"  # optional hard stop
```

- **interval** uses `every_seconds` / `every_minutes` / `every_hours`.
- **once** uses `at_time: "2026-11-01T09:00:00"` (or a wall-clock `at`).
- **days: [mon, tue, ...]** restricts a daily schedule to weekdays.
- **catch_up**: `run` runs a late occurrence once on return; `skip` records it
  missed (honestly, never pretended); `coalesce` collapses many missed into one.

Schedules are DST-correct via `zoneinfo`; where zone data is missing the runtime
falls back to a fixed UTC offset and is honest about the loss of DST.

Change schedules by editing this file and restarting the runtime. Inspect them
at runtime with `/schedules`.

## 5. Configure limits, quiet hours, and digests

Notification limits live under `notifications:` (all configurable; the shipped
defaults are conservative):

```yaml
notifications:
  timezone: "America/New_York"
  interval_seconds: 28800     # min spacing between standalone messages (8h)
  max_per_interval: 1         # <=1 unsolicited standalone per interval
  min_gap_seconds: 28800      # <=1 unsolicited standalone per 8h
  max_per_day: 2              # per local day; a digest counts toward this
  min_importance: 0.55        # below this -> digest, not an immediate message
  high_priority_importance: 0.85
  allow_high_priority_bypass: true
  quiet_hours: { start: "22:30", end: "07:30" }   # wraps midnight
  digest_enabled: true
  digest_at: "18:00"
```

Behaviour: an outcome above `high_priority_importance` may bypass the *interval*
limits but never an explicit pause/mute or quiet hours. Everything else either
delivers within the limits, is grouped into the scheduled digest, or is dropped.

### Adaptation to non-response

If several proactive messages go unanswered, Astra raises the bar for the next
ones (she does not stop, and she never treats silence as rejection). This is a
separate, content-free layer (`astra/engagement.py`; counters and timestamps
only, never message content) that can only *raise* the importance threshold - it
never relaxes a limit, quiet hours, or a pause:

```yaml
engagement:
  enabled: true
  reply_window_seconds: 21600   # a reply must arrive within this to count
  silence_after: 2              # unanswered messages before the bar rises
  threshold_step: 0.08          # extra threshold per further unanswered message
  max_threshold_boost: 0.25     # cap on the raise
  decay_days: 3.0               # old silence fades; time alone recovers her
```

A delivered message opens an obligation; a human message (from the console or
Discord) within the window discharges it. A message arriving after the window is
treated as unrelated and is neither counted as engagement nor accrued as
silence, so an away period does not penalise her. She always recovers: the pull
decays with time, and a reply clears it. Inspect it in `/autonomy`.

## 6. Pause, resume, and inspect (CLI / Discord)

These commands work from the console or Discord and never call the model:

| Command | Effect |
|---|---|
| `/autonomy` | Truthful status: paused?, last wake, notifications, scheduled activities, last completed. |
| `/autonomy pause` / `resume` | Pause/resume **activity execution** only. |
| `/autonomy now` | Run one bounded wake on demand (for verification). |
| `/notifications` | Status (sent today/interval, quiet hours, pending digest). |
| `/notifications pause` / `resume` | Pause/resume **proactive messaging** only. |
| `/notifications mute` / `unmute` | Silence notifications without stopping work. |
| `/schedules` | List configured/scheduled activities and their due times. |
| `/discord` | Connection state, last poll, last proactive send, delivery counters, last error. |
| `/power` (or `/machine`) | Measured machine resources (cpu/memory/uptime), efficiency mode, and suspend history. |
| `/proposals` | Improvement proposals awaiting your acknowledgement; `ack`/`approve`/`decline <id>`. |
| `/tasks` | Task-store counts and non-terminal tasks. |
| `/sleep-log` | Recent attempts and outcomes (the ledger). |

Activity and notifications are **independent**: pausing work does not silence
messaging, and silencing messaging does not stop work.

Approving a proposal **records a decision only**; it never applies a change to
identity, boundaries, credentials, permissions or source code. Those scopes are
flagged `requires_approval` and are applied by a separate, deliberate step, not
by the runtime.

## 7. Machine awareness and power behaviour

`astra/machine.py` samples the real machine - CPU load, memory pressure, uptime,
idle time - stdlib-first (`getloadavg`, `/proc`, `os.uptime`), with `psutil`
used only if installed. **An unmeasurable value is `None` and changes no
decision**, so absence never invents a fact.

Two behaviours follow from it, both automatic (no command needed):

- **Efficiency mode.** Heavy work is admitted only when the machine is *warm*
  (up long enough) and *sustainably calm* (the calm must last a sample window,
  so a momentary spike does not admit heavy work). Otherwise the scheduler runs
  only light kinds (`REASON_MACHINE_COLD`).
- **Sleeping alongside the computer.** A suspend is detected from the machine's
  own uptime versus the process clock. On a detected suspend Astra moves to
  SLEEP, so the gap is not mistaken for active time and scheduled work is not
  run through a period the computer was off. On resume, work stays conservative
  until the machine is calm again; waking back to ACTIVE is opt-in
  (`runtime.power.wake_on_resume`, default false).

## 8. Emotions are a cause of behaviour, not just a report

Astra's current affect is fed through `astra/behavior.py` into what she actually
does:

- **Which work runs.** The scheduler admits only light kinds when her state is
  flat or sad (`REASON_LOW_ENERGY`, `LOW_ENERGY_THRESHOLD`), via an injected
  `intensity` provider. A glad, curious state admits intensive kinds.
- **How much she takes on.** `ActivityPlanner` consumes the same modulation, so
  a deep low mood takes on no new work this pass and honestly reports `idle`; a
  glad state takes on up to its bound. This is the state biasing selection, not
  a mood the model narrates.
- **How she reads.** The reader gate folds the modulation's reading modifiers
  into `reading.should_read`, so a sad/weary state can genuinely stand reading
  *down*, not merely slow it.

All of it is derived, read-only, and stores nothing new; a neutral state changes
nothing. Nothing numeric leaks into the prompt (asserted by the behavioural
suite).

### How she talks about her plans

Astra's own plans are expressed from the same records the systems already write,
so what she says and what is recorded cannot drift. `astra/intentions.py` is a
pure, read-only view over the task store, the planner's evidence-backed
candidates and the attempt ledger. When a turn asks about her plans
("what are you planning?", "what are you working on?"), the orchestrator renders
`ASTRA'S OWN PLANS (HER INTENTIONS - NOT A SYSTEM REPORT)`, in the first person,
keeping the categories distinct:

- **committed** - scheduled/recurring work (a standing intention, not a request);
- **chosen for herself** - work selected by the bounded planner;
- **occupied with** - the task actually running right now;
- **postponed / blocked / waiting** and **meant to retry**;
- **only considering** - candidates she has evidence for but has *not* decided
  (never presented as a decision);
- **recently** - what she finished or set aside, from the ledger.

Her affect colours the framing through the *same* `behavior.activity_bias` the
planner uses to choose work, so a low state sounds reserved and leans to lighter
plans, a warm one forward - and a neutral state adds nothing. The view is
attached read-only in `build_session` (`set_intentions_source` ->
`RuntimeController.snapshot_intentions`). Asking about a plan never starts,
finishes or reorders it: `build_snapshot` calls only read accessors, and every
state transition stays in the task/activity lifecycle. When nothing is planned
she says so plainly rather than inventing an activity.

## 9. Verify the system

```bash
python -m tests                                  # full suite
python -m unittest tests.test_schedules -v       # time arithmetic
python -m unittest tests.test_schedule_runtime -v # materialise/refresh/re-arm
python -m unittest tests.test_activity -v        # bounded planning
python -m unittest tests.test_autonomy -v        # wake + proactive lifecycle
python -m unittest tests.test_notifications -v   # limits/quiet hours/dedupe
python -m unittest tests.test_reply_policy -v    # reply/ping convention
python -m unittest tests.test_intentions -v      # plans-as-intentions view
python -m unittest tests.test_discord_adapter -v # transport + delivery bookkeeping
python -m unittest tests.test_machine -v         # real machine awareness + suspend
python -m unittest tests.test_affect_propagation -v # emotions cause work/reading
python -m unittest tests.test_power_runtime -v   # power/suspend at runtime
python -m unittest tests.test_proposals -v       # approval boundary
python -m unittest tests.test_engagement -v      # non-response adaptation + reading outcomes
```

Manual end-to-end check:

1. Set `DISCORD_BOT_TOKEN` and `discord.channel_id`, start `python main.py --headless`.
2. From the console (if also open) or via the Discord channel, run `/discord`.
   `connected: True` and a recent `last successful poll` confirm the transport.
3. Run `/autonomy now` to force one wake; confirm the report matches the ledger.
4. Set a near-future schedule (e.g. an `interval` of a few minutes), wait, and
   confirm the activity runs and a message is delivered within the limits. If it
   does not, inspect `/discord` for a delivery error and `/sleep-log` for the
   recorded outcome.

## 8. Honest limitations

- **Model availability.** If Ollama is unreachable, model-dependent work is not
  marked complete; it fails or defers and is retried within bounds. A missed
  schedule is recorded as missed, never pretended. In particular the background
  reader records an unreachable model as a `FAILED` activity (retried a bounded
  number of times, then blocked) and holds its position so no passage is
  skipped; a courtesy hold (Roum active, a game, load, low concentration) is
  recorded as `SKIPPED`, and only a real read is `COMPLETED`.
- **Background reading is enabled for release.** `config/runtime.yaml`
  `model_use.reading: true` turns on the one autonomous model call. It runs only
  when a book is queued and Roum is idle; set it to `false` to stop it (an
  explicit `/read` still works, and the deterministic `Library` is always built).
- **Affect decay** is lazy (documented in `AGENTS.md`); it is unrelated to
  scheduling.
- **REST-only presence.** The adapter uses Discord's REST API; it cannot change
  gateway presence. That is a display nicety, not a delivery concern.
- **Autonomy is bounded**, not unrestricted: one wake does a small, fixed
  amount of work, and a planner that could generate work forever is deliberately
  prevented. Idle is a legitimate, reported outcome.

## 9. The reply/ping convention

Astra does **not** reply-attach by default. She attaches a reply (and pings the
message's author) only when she deliberately wants to underline one specific
message's importance, which she signals by beginning her response with the
marker defined in `astra/reply_policy.py` (`EMPHASIS_MARKER`). The adapter
strips the marker, attaches the reply to that message, and includes the mention
token. This is the single code-owned place that decides; the generation contract
(`astra/generation.py`) tells her when to use it and that she should not
otherwise.
