#!/usr/bin/env python3
"""
End-to-end test for bot memory system
"""
import os
import sys
import tempfile
import asyncio
from loguru import logger

sys.path.insert(0, '.')

async def test_bot_memory():
    """Test the complete bot memory pipeline"""

    # Setup temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['HOTMEM_SQLITE'] = os.path.join(tmpdir, 'test.db')
        os.environ['HOTMEM_LMDB_DIR'] = os.path.join(tmpdir, 'test.lmdb')
        os.environ['USER_ID'] = 'test_user'

        # Import after setting env vars
        from components.memory.memory_store import MemoryStore, Paths
        from components.memory.hotmemory_facade import HotMemoryFacade
        from components.processing.hotpath_processor import HotPathMemoryProcessor
        from pipecat.frames.frames import TranscriptionFrame
        from pipecat.processors.frame_processor import FrameDirection

        # Create store
        paths = Paths(
            sqlite_path=os.environ['HOTMEM_SQLITE'],
            lmdb_dir=os.environ['HOTMEM_LMDB_DIR']
        )
        store = MemoryStore(paths)

        # Create mock context aggregator
        class MockContextAggregator:
            def __init__(self):
                self.messages = []

            def user(self):
                return self

            @property
            def context(self):
                return self

            def add_message(self, msg):
                self.messages.append(msg)
                print(f"  [Context] Added message: {msg['role']}")
                if 'Memory Context:' in msg.get('content', ''):
                    # Extract bullets
                    content = msg['content']
                    if 'Memory Context:' in content:
                        memory_part = content.split('Memory Context:')[1]
                        if 'Memory Guidance:' in memory_part:
                            memory_part = memory_part.split('Memory Guidance:')[0]
                        bullets = [b.strip() for b in memory_part.strip().split('\n') if b.strip()]
                        for bullet in bullets[:3]:  # Show first 3 bullets
                            print(f"    - {bullet}")

        # Create processor with mock aggregator
        aggregator = MockContextAggregator()
        processor = HotPathMemoryProcessor(
            sqlite_path=os.environ['HOTMEM_SQLITE'],
            lmdb_dir=os.environ['HOTMEM_LMDB_DIR'],
            user_id='test_user',
            context_aggregator=aggregator
        )

        print("=== Bot Memory End-to-End Test ===\n")

        # Initialize the processor with StartFrame
        from pipecat.frames.frames import StartFrame
        start_frame = StartFrame()
        await processor.process_frame(start_frame, FrameDirection.DOWNSTREAM)

        # Test 1: Process fact statement
        print("1. Processing: 'My dog Potola is 5 years old'")
        frame1 = TranscriptionFrame(text="My dog Potola is 5 years old", user_id="test_user", timestamp="2025-01-01T00:00:00")
        frame1.is_final = True  # Set as attribute
        await processor.process_frame(frame1, FrameDirection.DOWNSTREAM)
        print()

        # Test 2: Query (should retrieve and inject memory)
        print("2. Processing: 'How old is my dog?'")
        aggregator.messages.clear()  # Clear previous messages
        frame2 = TranscriptionFrame(text="How old is my dog?", user_id="test_user", timestamp="2025-01-01T00:00:01")
        frame2.is_final = True
        await processor.process_frame(frame2, FrameDirection.DOWNSTREAM)
        print()

        # Test 3: Another fact
        print("3. Processing: 'Potola loves playing fetch'")
        aggregator.messages.clear()
        frame3 = TranscriptionFrame(text="Potola loves playing fetch", user_id="test_user", timestamp="2025-01-01T00:00:02")
        frame3.is_final = True
        await processor.process_frame(frame3, FrameDirection.DOWNSTREAM)
        print()

        # Test 4: Query about Potola
        print("4. Processing: 'Tell me about Potola'")
        aggregator.messages.clear()
        frame4 = TranscriptionFrame(text="Tell me about Potola", user_id="test_user", timestamp="2025-01-01T00:00:03")
        frame4.is_final = True
        await processor.process_frame(frame4, FrameDirection.DOWNSTREAM)
        print()

        # Test 5: Correction
        print("5. Processing: 'Actually, Potola is 7 years old'")
        aggregator.messages.clear()
        frame5 = TranscriptionFrame(text="Actually, Potola is 7 years old", user_id="test_user", timestamp="2025-01-01T00:00:04")
        frame5.is_final = True
        await processor.process_frame(frame5, FrameDirection.DOWNSTREAM)
        print()

        # Test 6: Query again to see if correction worked
        print("6. Processing: 'How old is Potola now?'")
        aggregator.messages.clear()
        frame6 = TranscriptionFrame(text="How old is Potola now?", user_id="test_user", timestamp="2025-01-01T00:00:05")
        frame6.is_final = True
        await processor.process_frame(frame6, FrameDirection.DOWNSTREAM)
        print()

        print("✅ Test complete!")

if __name__ == "__main__":
    # Suppress most logs
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    asyncio.run(test_bot_memory())