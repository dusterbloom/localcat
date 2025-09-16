#!/usr/bin/env python3
"""
Test the fixed performance optimizations
"""

import time
import os
import sys

# Add server path and activate environment
sys.path.insert(0, '/Users/peppi/Dev/localcat/server')
os.chdir('/Users/peppi/Dev/localcat/server')

def test_intent_classification_speed():
    """Test the speed of intent classification"""
    print("🚀 Testing Intent Classification Speed")
    print("=" * 50)

    # Import the fixed classifier
    from components.memory.memory_intent import get_intent_classifier

    classifier = get_intent_classifier()

    test_cases = [
        "Good evening Socrates, can you hear me?",
        "Hello there!",
        "What is the weather like?",
        "My dog's name is Max",
        "OK got it",
        "Wow that's amazing!",
        "Actually, I meant something else",
        "Can you help me?",
        "Goodbye!"
    ]

    times = []

    for text in test_cases:
        start = time.perf_counter()
        result = classifier.analyze(text)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        skip_str = "SKIP" if not result.requires_retrieval else "RETR"
        store_str = "STORE" if result.requires_memory else "SKIP"

        print(f"{elapsed:6.2f}ms | {result.intent.value:15} | {skip_str:4} | {store_str:5} | '{text[:40]}...'")

    avg_time = sum(times) / len(times)
    max_time = max(times)

    print(f"\n📊 PERFORMANCE RESULTS:")
    print(f"   Average: {avg_time:.2f}ms")
    print(f"   Maximum: {max_time:.2f}ms")
    print(f"   Target:  <10ms per classification")

    if avg_time < 10:
        print(f"✅ PERFORMANCE TARGET MET!")
    else:
        print(f"❌ Still too slow, needs more optimization")

    return avg_time

def test_session_storage_speed():
    """Test session storage operations"""
    print("\n🗄️ Testing Session Storage Speed")
    print("=" * 50)

    from components.session.session_store import SessionStore

    session_store = SessionStore()

    # Test session creation
    start = time.perf_counter()
    session = session_store.create_session("test_user", "test_agent")
    session_time = (time.perf_counter() - start) * 1000

    # Test message addition
    start = time.perf_counter()
    session_store.add_message(session.session_id, "user", "Test message")
    message_time = (time.perf_counter() - start) * 1000

    print(f"Session creation: {session_time:.2f}ms")
    print(f"Message addition: {message_time:.2f}ms")
    print(f"Total DB ops:     {session_time + message_time:.2f}ms")

    if session_time + message_time < 50:
        print(f"✅ SESSION STORAGE IS FAST")
    else:
        print(f"⚠️ Session storage needs optimization")

    return session_time + message_time

def main():
    """Run all performance tests"""
    print("🔥 LOCALCAT PERFORMANCE FIX VALIDATION")
    print("=" * 60)

    intent_time = test_intent_classification_speed()
    db_time = test_session_storage_speed()

    total_expected = intent_time + db_time + 50  # 50ms for other overhead

    print(f"\n🎯 EXPECTED PIPELINE PERFORMANCE:")
    print(f"   Intent classification: {intent_time:.1f}ms")
    print(f"   Database operations:   {db_time:.1f}ms")
    print(f"   Other overhead:        ~50ms")
    print(f"   TOTAL EXPECTED:        {total_expected:.1f}ms")

    if total_expected < 200:
        print(f"\n🚀 PIPELINE SHOULD BE FAST NOW!")
        print(f"   Previous: 712ms → New: {total_expected:.1f}ms")
        print(f"   Improvement: {712 - total_expected:.1f}ms saved ({((712 - total_expected) / 712 * 100):.0f}% faster)")
    else:
        print(f"\n⚠️ Still needs more optimization")

if __name__ == "__main__":
    main()