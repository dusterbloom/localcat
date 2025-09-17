#!/usr/bin/env python3
"""
Interactive Memory REPL (Read-Eval-Print Loop) for LocalCat
============================================================

Test memory operations interactively:
- Store facts
- Query and retrieve
- See intent classification
- Debug memory behavior

Usage:
    python memory_repl.py [--db memory.db] [--lmdb graph.lmdb]
"""

import os
import sys
import tempfile
import readline  # For better input experience
from typing import Optional
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent))

from components.memory.memory_store import MemoryStore, Paths
from components.memory.hotmemory_facade import HotMemoryFacade
from components.memory.rule_v2_adapter import RuleV2Adapter
from loguru import logger

# Reduce logging noise
logger.remove()
logger.add(sys.stderr, level="WARNING")


class MemoryREPL:
    """Interactive memory testing REPL"""

    def __init__(self, db_path: Optional[str] = None, lmdb_path: Optional[str] = None):
        """Initialize with optional database paths or use temp files"""

        if db_path and lmdb_path:
            # Use provided paths
            self.temp_dir = None
            self.paths = Paths(sqlite_path=db_path, lmdb_dir=lmdb_path)
            print(f"📁 Using database: {db_path}")
        else:
            # Create temp directory
            self.temp_dir = tempfile.TemporaryDirectory()
            self.paths = Paths(
                sqlite_path=os.path.join(self.temp_dir.name, 'test.db'),
                lmdb_dir=os.path.join(self.temp_dir.name, 'test.lmdb')
            )
            print(f"📁 Using temporary database in {self.temp_dir.name}")

        # Initialize memory system
        self.store = MemoryStore(self.paths)
        self.memory = HotMemoryFacade(self.store)
        self.classifier = RuleV2Adapter()

        # Session tracking
        self.session_id = "repl_session"
        self.turn_id = 0
        self.user_id = "test_user"

        # Print welcome
        self.print_welcome()

    def print_welcome(self):
        """Print welcome message and help"""
        print("\n" + "="*60)
        print("🧠 LocalCat Memory REPL")
        print("="*60)
        print("\nCommands:")
        print("  /help          - Show this help")
        print("  /store <text>  - Store a fact")
        print("  /query <text>  - Query memory")
        print("  /intent <text> - Check intent classification")
        print("  /stats         - Show memory statistics")
        print("  /clear         - Clear all memory")
        print("  /exit          - Exit REPL")
        print("\nOr just type text to see full processing pipeline")
        print("-"*60)

    def process_command(self, text: str) -> bool:
        """Process commands. Returns False to exit."""

        if text == "/exit":
            return False

        elif text == "/help":
            self.print_welcome()

        elif text == "/clear":
            # Reset memory
            if self.temp_dir:
                print("♻️  Memory cleared (temp DB reset)")
            else:
                print("⚠️  Cannot clear persistent database")

        elif text == "/stats":
            self.show_stats()

        elif text.startswith("/store "):
            fact = text[7:].strip()
            if fact:
                self.store_fact(fact)
            else:
                print("❌ Usage: /store <fact>")

        elif text.startswith("/query "):
            query = text[7:].strip()
            if query:
                self.query_memory(query)
            else:
                print("❌ Usage: /query <text>")

        elif text.startswith("/intent "):
            text_to_classify = text[8:].strip()
            if text_to_classify:
                self.check_intent(text_to_classify)
            else:
                print("❌ Usage: /intent <text>")

        elif text.startswith("/"):
            print(f"❌ Unknown command: {text}")
            print("   Type /help for available commands")

        else:
            # Process as regular text through full pipeline
            self.process_text(text)

        return True

    def store_fact(self, text: str):
        """Store a fact explicitly"""
        print(f"\n📝 Storing: '{text}'")

        self.turn_id += 1
        result = self.memory.process_turn(text, self.session_id, self.turn_id, self.user_id)

        if result.triples:
            print(f"✅ Stored {len(result.triples)} triple(s):")
            for s, r, d in result.triples:
                print(f"   ({s}, {r}, {d})")
        else:
            print("⚠️  No triples extracted")

    def query_memory(self, text: str):
        """Query memory explicitly"""
        print(f"\n🔍 Querying: '{text}'")

        self.turn_id += 1
        result = self.memory.process_turn(text, self.session_id, self.turn_id, self.user_id)

        print(f"Intent: {result.intent.intent.value}")
        print(f"Needs retrieval: {result.needs_retrieval}")

        if result.bullets:
            print(f"\n📋 Retrieved {len(result.bullets)} bullet(s):")
            for bullet in result.bullets:
                print(f"   • {bullet}")
        else:
            print("⚠️  No bullets retrieved")

    def check_intent(self, text: str):
        """Check intent classification"""
        print(f"\n🏷️  Classifying: '{text}'")

        result = self.classifier.classify(text)

        print(f"Intent: {result.intent.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Requires retrieval: {result.requires_retrieval}")
        print(f"Requires memory storage: {result.requires_memory}")

    def process_text(self, text: str):
        """Process text through full pipeline"""
        print(f"\n🔄 Processing: '{text}'")
        print("-"*40)

        # 1. Intent classification
        intent = self.classifier.classify(text)
        print(f"1️⃣  Intent: {intent.intent.value} (confidence: {intent.confidence:.2f})")
        print(f"   Retrieval: {intent.requires_retrieval}, Storage: {intent.requires_memory}")

        # 2. Memory processing
        self.turn_id += 1
        result = self.memory.process_turn(text, self.session_id, self.turn_id, self.user_id)

        # 3. Show extraction
        if result.triples:
            print(f"\n2️⃣  Extracted {len(result.triples)} triple(s):")
            for s, r, d in result.triples:
                print(f"   ({s}, {r}, {d})")
        else:
            print("\n2️⃣  No triples extracted")

        # 4. Show retrieval
        if result.bullets:
            print(f"\n3️⃣  Retrieved {len(result.bullets)} bullet(s):")
            for bullet in result.bullets[:5]:  # Show max 5
                print(f"   • {bullet}")
            if len(result.bullets) > 5:
                print(f"   ... and {len(result.bullets) - 5} more")
        else:
            print("\n3️⃣  No bullets retrieved")

    def show_stats(self):
        """Show memory statistics"""
        print("\n📊 Memory Statistics")
        print("-"*40)

        # Count entities
        entity_count = len(self.memory.entity_index)
        edge_count = sum(len(edges) for edges in self.memory.entity_index.values())

        print(f"Entities: {entity_count}")
        print(f"Edges: {edge_count}")
        print(f"Session: {self.session_id}")
        print(f"Turns: {self.turn_id}")

        # Show some entities
        if entity_count > 0:
            print(f"\nSample entities:")
            for entity in list(self.memory.entity_index.keys())[:5]:
                print(f"  - {entity}")

    def run(self):
        """Run the REPL loop"""
        while True:
            try:
                # Get input
                text = input("\n🧠> ").strip()

                if not text:
                    continue

                # Process command or text
                if not self.process_command(text):
                    break

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

    def __del__(self):
        """Cleanup temp directory if used"""
        if hasattr(self, 'temp_dir') and self.temp_dir:
            self.temp_dir.cleanup()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="LocalCat Memory REPL")
    parser.add_argument("--db", help="SQLite database path (default: temp)")
    parser.add_argument("--lmdb", help="LMDB directory path (default: temp)")

    args = parser.parse_args()

    # Create and run REPL
    repl = MemoryREPL(args.db, args.lmdb)
    repl.run()


if __name__ == "__main__":
    main()