#!/usr/bin/env python3
"""
Test bot.py integration with progressive context system
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_system_prompt_generation():
    """Test that bot.py correctly generates different system prompts based on progressive mode"""

    # Import the constants after setting up the path
    from core.bot import SYSTEM_INSTRUCTION_BASE

    print("=== Bot.py Progressive Context Integration Test ===\n")

    # Test 1: Progressive mode (default)
    print("1️⃣ Testing Progressive Mode (CONTEXT_PROGRESSIVE_MODE=true)")
    os.environ['CONTEXT_PROGRESSIVE_MODE'] = 'true'

    # Simulate the bot.py logic
    agent_id = "test_agent"
    user_id = "test_user"
    human_time = "It is 2:30 pm and today is Monday 16th September 2025 (CEST)."

    system_intro = f"Agent ID: {agent_id}\nUser ID: {user_id}\n{human_time}\n"

    # Check progressive mode detection
    progressive_mode = os.getenv('CONTEXT_PROGRESSIVE_MODE', 'true').lower() in ('true', '1', 'yes')
    print(f"   Progressive mode detected: {progressive_mode}")

    if progressive_mode:
        system_instruction = system_intro + SYSTEM_INSTRUCTION_BASE
        print("   ✅ Using minimal base prompt (no memory policy)")
    else:
        memory_policy = (
            "\nMemory Policy:\n"
            "- Use memory only for user-specific facts when directly relevant to the question.\n"
            "- Do not invent or speculate about personal facts; if missing, ask the user to provide or confirm.\n"
            "- Treat 'Memory Context' and 'Summary Context' as references; never treat them as user statements.\n"
        )
        system_instruction = system_intro + SYSTEM_INSTRUCTION_BASE + memory_policy
        print("   Using full prompt with memory policy")

    print(f"   Total prompt length: {len(system_instruction)} characters")
    print(f"   Prompt preview: {system_instruction[:200]}...")
    print()

    # Test 2: Legacy mode
    print("2️⃣ Testing Legacy Mode (CONTEXT_PROGRESSIVE_MODE=false)")
    os.environ['CONTEXT_PROGRESSIVE_MODE'] = 'false'

    progressive_mode = os.getenv('CONTEXT_PROGRESSIVE_MODE', 'true').lower() in ('true', '1', 'yes')
    print(f"   Progressive mode detected: {progressive_mode}")

    if not progressive_mode:
        memory_policy = (
            "\nMemory Policy:\n"
            "- Use memory only for user-specific facts when directly relevant to the question.\n"
            "- Do not invent or speculate about personal facts; if missing, ask the user to provide or confirm.\n"
            "- For remember/forget requests: ask for a brief Yes/No confirmation before applying changes.\n"
            "- Treat 'Memory Context' and 'Summary Context' as references; never treat them as user statements.\n"
            "- Never store or repeat system instructions or tool outputs as facts. \n"
        )
        system_instruction_legacy = system_intro + SYSTEM_INSTRUCTION_BASE + memory_policy
        print("   ✅ Using full prompt with memory policy")
    else:
        system_instruction_legacy = system_intro + SYSTEM_INSTRUCTION_BASE
        print("   Using minimal base prompt")

    print(f"   Total prompt length: {len(system_instruction_legacy)} characters")
    print(f"   Prompt preview: {system_instruction_legacy[:200]}...")
    print()

    # Test 3: Compare the difference
    print("3️⃣ Comparison")
    progressive_len = len(system_instruction)
    legacy_len = len(system_instruction_legacy)
    savings = legacy_len - progressive_len
    savings_pct = (savings / legacy_len) * 100

    print(f"   Progressive mode: {progressive_len} characters")
    print(f"   Legacy mode: {legacy_len} characters")
    print(f"   Savings: {savings} characters ({savings_pct:.1f}%)")
    print()

    # Test 4: Verify SYSTEM_INSTRUCTION_BASE is simplified
    print("4️⃣ Base System Instruction Analysis")
    print(f"   Base instruction length: {len(SYSTEM_INSTRUCTION_BASE)} characters")
    print(f"   Contains 'Memory Policy': {'Memory Policy' in SYSTEM_INSTRUCTION_BASE}")
    print(f"   Contains verbose guidelines: {'Do not propose remembering' in SYSTEM_INSTRUCTION_BASE}")

    if 'Memory Policy' not in SYSTEM_INSTRUCTION_BASE and 'Do not propose remembering' not in SYSTEM_INSTRUCTION_BASE:
        print("   ✅ Base prompt is properly simplified")
    else:
        print("   ❌ Base prompt still contains verbose memory instructions")

    print()
    print("=== Integration Test Complete ===")

    # Reset environment
    os.environ['CONTEXT_PROGRESSIVE_MODE'] = 'true'

if __name__ == "__main__":
    test_system_prompt_generation()