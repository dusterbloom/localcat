#!/usr/bin/env python3
"""
Fix for Pipecat LLMTextFrame duplication bug.

BUG: In LLMContextResponseAggregator.process_frame(), LLMTextFrames are being
pushed downstream with await self.push_frame(frame, direction). This causes
LLMTextFrames to reach the TTS pipeline, resulting in duplicate audio.

SOLUTION: LLMTextFrames should be consumed by the context aggregator for
building conversation context only, and should NOT be pushed downstream to TTS.

This script patches the bug by modifying the LLMContextResponseAggregator
process_frame method to NOT push LLMTextFrames downstream.
"""

import os
import sys

def patch_llm_context_aggregator():
    """Apply patch to fix LLMTextFrame duplication bug."""

    # Find the file to patch
    target_file = None
    for path in sys.path:
        potential_path = os.path.join(path, "pipecat/processors/aggregators/llm_response.py")
        if os.path.exists(potential_path):
            target_file = potential_path
            break

    if not target_file:
        print("❌ Could not find llm_response.py to patch")
        return False

    print(f"🎯 Found file to patch: {target_file}")

    # Read the original file
    with open(target_file, 'r') as f:
        content = f.read()

    # Check if already patched
    if "# PATCHED: Don't push LLMTextFrames downstream" in content:
        print("✅ File already patched")
        return True

    # Find and replace the problematic code
    original_code = '''        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._handle_llm_start(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end(frame)
        elif isinstance(frame, LLMTextFrame):
            await self._handle_llm_text(frame)
        await self.push_frame(frame, direction)'''

    patched_code = '''        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._handle_llm_start(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end(frame)
        elif isinstance(frame, LLMTextFrame):
            await self._handle_llm_text(frame)
            # PATCHED: Don't push LLMTextFrames downstream - they're for context only
            # LLMTextFrames should not reach TTS processing pipeline
            return
        await self.push_frame(frame, direction)'''

    if original_code not in content:
        print("❌ Could not find the exact code to patch")
        print("Looking for pattern...")

        # Try a more flexible search
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'elif isinstance(frame, LLMTextFrame):' in line:
                print(f"Found LLMTextFrame handling at line {i+1}")
                # Show surrounding context
                for j in range(max(0, i-3), min(len(lines), i+8)):
                    marker = ">>> " if j == i else "    "
                    print(f"{marker}{j+1:3d}: {lines[j]}")
                return False

        return False

    # Apply the patch
    new_content = content.replace(original_code, patched_code)

    # Write the patched file
    with open(target_file, 'w') as f:
        f.write(new_content)

    print("✅ Successfully patched LLMContextResponseAggregator")
    print("🔧 LLMTextFrames will no longer be pushed downstream to TTS")

    return True

def main():
    """Main patch application function."""
    print("🚀 APPLYING PIPECAT LLMTEXTFRAME DUPLICATION FIX")
    print("=" * 60)

    success = patch_llm_context_aggregator()

    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: LLMTextFrame duplication bug has been fixed!")
        print("   - LLMTextFrames will be consumed by context aggregator only")
        print("   - No more duplicate TextFrames reaching TTS pipeline")
        print("   - TTS duplication should be eliminated")
        print("\n⚠️  RESTART THE SERVER TO APPLY THE FIX")
    else:
        print("❌ FAILURE: Could not apply the patch")
        print("   - File may have been modified or already patched")
        print("   - Check the llm_response.py file manually")
    print("=" * 60)

if __name__ == "__main__":
    main()