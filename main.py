import sys
from datetime import datetime, timezone
from memory_store import TripleMemoryStore
from orchestrator import CompanionOrchestrator
from consolidator import consolidate_turn

def print_help():
    print("\n--- CLI COMMANDS ---")
    print(" /memories [target] : List memories with key metadata (target optional: roum, self, relationship)")
    print(" /memory <id>       : Display full record and provenance for a memory")
    print(" /correct           : Interactive workflow to supersede an incorrect memory")
    print(" /journal           : Display AI journal reflections")
    print(" /exit              : Save session and exit")
    print("--------------------\n")

def display_memories(store: TripleMemoryStore, target_filter: str = None):
    targets = [target_filter] if target_filter in ["roum", "self", "relationship"] else ["roum", "self", "relationship"]
    
    for t in targets:
        print(f"\n=== {t.upper()} MODEL MEMORIES ===")
        mems = store._read_file(t)
        if not mems:
            print("  (No memories found)")
            continue
        for m in mems:
            status_indicator = "ACTIVE" if m.get("status") == "active" else f"SUPERSEDED by {m.get('superseded_by')}"
            print(f" ID: {m['id']} | [{m['type']}] | Status: {status_indicator}")
            print(f"    Content: {m['content']}")
            print(f"    Source: {m['source']} | Conf: {m['confidence']} | Turn: {m['originating_turn']}")
            print("-" * 50)

def display_full_memory(store: TripleMemoryStore, mem_id: str):
    found = store.get_memory_by_id(mem_id)
    if not found:
        print(f"\nError: Memory ID '{mem_id}' not found in any model.")
        return
    
    target, mem = found
    print(f"\n=== FULL RECORD: {mem['id']} ({target.upper()} MODEL) ===")
    print(json.dumps(mem, indent=2, ensure_ascii=False))

def run_correction_workflow(store: TripleMemoryStore, conv_id: str, turn: int):
    print("\n--- MEMORY CORRECTION WORKFLOW ---")
    mem_id = input("Enter Memory ID to correct: ").strip()
    found = store.get_memory_by_id(mem_id)
    
    if not found:
        print(f"Error: Memory ID '{mem_id}' not found.")
        return

    target, old_mem = found
    print(f"Current Content: \"{old_mem['content']}\"")
    new_content = input("Enter CORRECTED statement: ").strip()
    if not new_content:
        print("Correction cancelled (empty input).")
        return

    reason = input("Reason for correction (optional): ").strip() or "Explicit user correction"
    
    new_id = store.correct_memory(
        target_model=target,
        old_mem_id=mem_id,
        new_content=new_content,
        reason=reason,
        conversation_id=conv_id,
        turn=turn
    )
    
    print(f"\n✓ Memory corrected successfully!")
    print(f"  Old Memory '{mem_id}' -> Marked as 'superseded'")
    print(f"  New Memory '{new_id}' -> Marked as 'active'")

def main():
    store = TripleMemoryStore()
    orchestrator = CompanionOrchestrator(store)
    conv_id = f"conv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    history = []
    turn_counter = 0

    print("==========================================================")
    print("  Astra Local Companion Core (Gemma 4 E4B)")
    print("  Type /help for memory inspection & correction commands.")
    print("==========================================================")

    while True:
        try:
            user_input = input("\nRoum > ").strip()
            if not user_input:
                continue

            if user_input.lower() == "/help":
                print_help()
                continue

            if user_input.lower() in ["/exit", "exit", "quit"]:
                print("Session ended cleanly. Memories saved.")
                break

            if user_input.lower().startswith("/memories"):
                parts = user_input.split()
                target_filter = parts[1].lower() if len(parts) > 1 else None
                display_memories(store, target_filter)
                continue

            if user_input.lower().startswith("/memory"):
                parts = user_input.split()
                if len(parts) > 1:
                    display_full_memory(store, parts[1])
                else:
                    print("Usage: /memory <mem_id>")
                continue

            if user_input.lower() == "/correct":
                run_correction_workflow(store, conv_id, turn_counter)
                continue

            if user_input.lower() == "/journal":
                entries = store._read_file("journal")
                print("\n=== AI MEMORY JOURNAL ===")
                if not entries:
                    print("  (No journal reflections recorded yet)")
                for j in entries:
                    print(f"[{j['timestamp']}] {j['title']}")
                    print(f"  {j['observation']}\n")
                continue

            # Standard Conversation Turn
            turn_counter += 1
            prompt = orchestrator.build_prompt(user_input, history)
            response = orchestrator.query_gemma(prompt)

            print(f"\nAstra > {response}")

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

            # Log history turn and run post-turn consolidation
            store.log_history_turn(conv_id, turn_counter, user_input, response)
            consolidate_turn(user_input, response, store, conv_id, turn_counter)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            sys.exit(0)

if __name__ == "__main__":
    main()