import os
import time
from dotenv import load_dotenv

from memory_store import MemoryStore, Paths


def main():
    # Load server env if present
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)

    sqlite_path = os.getenv("HOTMEM_SQLITE", "memory.db")
    lmdb_dir = os.getenv("HOTMEM_LMDB_DIR", "graph.lmdb")
    print(f"Using HOTMEM_SQLITE={sqlite_path}\nUsing HOTMEM_LMDB_DIR={lmdb_dir}")

    store = MemoryStore(Paths(sqlite_path=sqlite_path, lmdb_dir=lmdb_dir))

    # Noisy edges to remove (s,r,d)
    removals = [
        ("morning", "quality", "good"),
        ("good evening", "quality", "good"),
        ("you", "do_with", "fumes"),
        ("it", "is", "what time"),
        ("you", "know", "what time it is"),
        ("is", "and", "stressing"),
        ("she", "stress", "you"),
        ("dog", "name", "potola and she is stressing me out"),
    ]

    # Apply removals
    for s, r, d in removals:
        try:
            print(f"Hard-forget: ({s}, {r}, {d})")
            store.hard_forget(s, r, d)
        except Exception as e:
            print(f"  Warning: failed to forget ({s},{r},{d}): {e}")

    # Optionally add a clean replacement for the dog name if relevant
    try:
        now_ts = int(time.time() * 1000)
        print("Observing clean edge: (dog, name, potola)")
        store.observe_edge("dog", "name", "potola", conf=0.8 if False else 0.8, now_ts=now_ts)  # type: ignore
    except TypeError:
        # Different signature in MemoryStore; fallback to correct call
        now_ts = int(time.time() * 1000)
        store.observe_edge("dog", "name", "potola", 0.8, now_ts)
    except Exception as e:
        print(f"  Warning: failed to observe clean dog name: {e}")

    # Ensure batched writes are flushed
    try:
        store.flush()
    except Exception:
        pass

    print("Cleanup complete.")


if __name__ == "__main__":
    main()

