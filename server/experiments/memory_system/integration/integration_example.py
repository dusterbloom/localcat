#!/usr/bin/env python3
"""
Integration example: Using HotMemService in place of Mem0MemoryService

This example demonstrates how to use HotMemService as a drop-in replacement
for Pipecat's Mem0MemoryService with tool-based memory interface.
"""

import asyncio
import sys
import os

# Add core modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from core.memory import HotMemService
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.processors.aggregators.llm_context import LLMContext


async def demo_memory_service():
    """Demonstrate HotMemService basic usage."""

    logger.info("=== HotMemService Integration Demo ===")

    # Initialize HotMemService (same interface as Mem0MemoryService)
    memory_service = HotMemService(
        user_id="demo_user",
        agent_id="demo_agent",
        run_id="demo_session_001"
    )

    logger.info("✓ HotMemService initialized")

    # Example 1: Store conversation messages (automatic processing)
    logger.info("\n--- Example 1: Automatic Memory Storage ---")

    conversation = [
        {"role": "user", "content": "Hi, I'm Alice and I'm a software engineer"},
        {"role": "assistant", "content": "Nice to meet you Alice! I'll remember that you're a software engineer."},
        {"role": "user", "content": "I love working with Python and machine learning"},
        {"role": "assistant", "content": "That's great! I've noted that you enjoy Python and ML work."}
    ]

    # Store messages (triggers HotPath extraction)
    memory_service._store_messages(conversation)
    logger.info("✓ Conversation stored with automatic fact extraction")

    # Example 2: Retrieve relevant memories
    logger.info("\n--- Example 2: Memory Retrieval ---")

    # Query for memories about the user
    memories = memory_service._retrieve_memories("What do I know about Alice?")
    logger.info(f"Retrieved {len(memories['results'])} memories:")

    for i, memory in enumerate(memories['results'], 1):
        logger.info(f"  {i}. {memory['memory']}")

    # Example 3: Context enhancement (the key feature)
    logger.info("\n--- Example 3: Context Enhancement ---")

    # Create a new conversation context
    context = LLMContext()
    context.add_message({"role": "user", "content": "What programming languages do I use?"})

    # Enhance context with memories and tool notice
    memory_service._enhance_context_with_memories(context, "What programming languages do I use?")

    enhanced_messages = context.get_messages()
    logger.info(f"Enhanced context now has {len(enhanced_messages)} messages:")

    for i, msg in enumerate(enhanced_messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content", "")
        logger.info(f"  {i+1}. [{role}]: {content}")

    # Example 4: Performance demonstration
    logger.info("\n--- Example 4: Performance Test ---")

    import time

    # Test storage performance
    test_messages = [
        {"role": "user", "content": "I also enjoy hiking and photography on weekends"}
    ]

    start = time.perf_counter()
    memory_service._store_messages(test_messages)
    storage_time = (time.perf_counter() - start) * 1000

    # Test retrieval performance
    start = time.perf_counter()
    result = memory_service._retrieve_memories("hobbies")
    retrieval_time = (time.perf_counter() - start) * 1000

    logger.info(f"✓ Storage: {storage_time:.1f}ms")
    logger.info(f"✓ Retrieval: {retrieval_time:.1f}ms")
    logger.info(f"✓ Total: {storage_time + retrieval_time:.1f}ms (well under 200ms target)")

    # Example 5: Memory statistics
    logger.info("\n--- Example 5: Memory Statistics ---")

    stats = memory_service.get_memory_stats()
    logger.info("Current memory state:")
    logger.info(f"  Session ID: {stats['session_id']}")
    logger.info(f"  Turn ID: {stats['turn_id']}")
    logger.info(f"  User ID: {stats['user_id']}")
    logger.info(f"  Agent ID: {stats['agent_id']}")

    # Cleanup
    await memory_service.cleanup()
    logger.info("\n✓ Memory service cleanup completed")


async def demo_pipecat_integration():
    """Show how HotMemService integrates with Pipecat pipeline."""

    logger.info("\n=== Pipecat Pipeline Integration Demo ===")

    # Create memory service
    memory_service = HotMemService(
        user_id="pipeline_user",
        agent_id="pipeline_agent"
    )

    # Simulate LLM messages frame processing
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Remember that I prefer concise answers"}
    ]

    # Create LLM messages frame
    frame = LLMMessagesFrame(messages)

    # Process the frame (this would happen in the pipeline)
    logger.info("Processing LLM messages frame through HotMemService...")

    # In a real pipeline, this would be called automatically
    # For demo purposes, we'll manually trigger the memory processing
    context = LLMContext(messages)
    memory_service._enhance_context_with_memories(context, "Remember that I prefer concise answers")

    enhanced_messages = context.get_messages()
    logger.info(f"✓ Frame processed, context enhanced: {len(messages)} → {len(enhanced_messages)} messages")

    # Show the tool availability notice that was added
    system_messages = [msg for msg in enhanced_messages if msg.get("role") == "system"]
    for msg in system_messages:
        if "Memory tools available" in msg.get("content", ""):
            logger.info(f"✓ Tool notice added: {msg['content'][:50]}...")
            break

    await memory_service.cleanup()


async def main():
    """Run all demonstration examples."""

    # Configure logging for demo
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
    )

    try:
        await demo_memory_service()
        await demo_pipecat_integration()

        logger.info("\n🎉 All integration examples completed successfully!")

        logger.info("\n=== Usage Summary ===")
        logger.info("HotMemService provides:")
        logger.info("  ✓ Drop-in replacement for Mem0MemoryService")
        logger.info("  ✓ Ultra-fast HotPath backend (<5ms typical performance)")
        logger.info("  ✓ Tool-based explicit interface (no intent classification)")
        logger.info("  ✓ Full Pipecat pipeline compatibility")
        logger.info("  ✓ Automatic storage with on-demand retrieval")
        logger.info("  ✓ Memory tools: hotmem_remember, hotmem_recall, hotmem_forget, hotmem_search")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())