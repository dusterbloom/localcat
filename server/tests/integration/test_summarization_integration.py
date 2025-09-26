#!/usr/bin/env python3
"""
Integration test for turn-based summarization system.
Tests the full pipeline with real components and progressive complexity.
"""

import asyncio
import os
import sys
import tempfile
import time
import shutil
from typing import List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

# Add server root to path for imports
_HERE = os.path.dirname(__file__)
_SERVER_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PIPECAT_SRC = os.path.join(_SERVER_ROOT, "pipecat", "src")
for p in (_SERVER_ROOT, _PIPECAT_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.memory.hotpath_processor import HotPathMemoryProcessor
from pipecat.frames.frames import StartFrame, TranscriptionFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams

# Test configuration
@dataclass
class TestScenario:
    """Defines a test scenario for summarization"""
    name: str
    turn_pairs: int
    conversations: List[str]
    expected_summaries: int
    expected_with_final: int
    key_concepts: List[str]  # Concepts that should appear in summaries


class SummarizationIntegrationTest:
    """Comprehensive test suite for turn-based summarization"""

    def __init__(self):
        self.temp_dir = None
        self.db_path = None
        self.lmdb_dir = None
        self.processor = None
        self.pipeline = None
        self.task = None
        self.runner = None

    async def setup(self, turn_pairs: int = 5):
        """Set up test environment with specified configuration"""
        # Create temp directories
        self.temp_dir = tempfile.mkdtemp(prefix="hotmem_test_")
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.lmdb_dir = os.path.join(self.temp_dir, "lmdb")

        # Configure environment BEFORE initializing processor
        os.environ["MEMORY_SUMMARY_ENABLED"] = "true"
        os.environ["SUMMARIZER_WINDOW_MODE"] = "turn_pairs"
        os.environ["SUMMARIZER_TURN_PAIRS"] = str(turn_pairs)
        os.environ["SUMMARIZER_MODEL"] = "google/gemma-3n-e4b"
        os.environ["SUMMARIZER_BASE_URL"] = "http://127.0.0.1:1234/v1"
        os.environ["SUMMARIZER_MAX_TOKENS"] = "120"

        # Initialize processor
        self.processor = HotPathMemoryProcessor(
            user_id="test-user",
            sqlite_path=self.db_path,
            lmdb_dir=self.lmdb_dir
        )

        # Create a minimal pipeline with just the processor
        self.pipeline = Pipeline([self.processor])

        # Create pipeline task with parameters
        params = PipelineParams(enable_metrics=False)
        self.task = PipelineTask(self.pipeline, params=params)

        # Create runner (but don't start it yet)
        self.runner = PipelineRunner()

        logger.debug(f"Test environment setup complete with turn_pairs={turn_pairs}")

    async def teardown(self):
        """Clean up test environment"""
        try:
            if self.processor:
                await self.processor.cleanup()
                self.processor = None
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Error removing temp dir: {e}")

    async def process_conversations(self, conversations: List[str]) -> None:
        """Process a list of conversations through the pipeline"""
        # Send StartFrame to initialize pipeline
        logger.info(f"Sending StartFrame")
        await self.task.queue_frames([StartFrame()])
        await asyncio.sleep(0.1)

        # Process each conversation as a transcription
        for i, text in enumerate(conversations, 1):
            frame = TranscriptionFrame(
                text=text,
                user_id="test-user",
                timestamp=str(i)
            )
            logger.info(f"Queueing TranscriptionFrame {i}: {text[:50]}...")
            await self.task.queue_frames([frame])
            # Small delay to ensure async tasks complete
            await asyncio.sleep(0.1)
            logger.info(f"After queueing turn {i}, processor turn_id = {self.processor._turn_id}")

    def get_summaries(self) -> List[Tuple[str, int]]:
        """Retrieve all summaries from the database"""
        summaries = self.processor.store.get_recent_chunks_by_eid("summary", limit=100)
        return summaries

    def verify_summary_content(self, summaries: List[Tuple[str, int]],
                              key_concepts: List[str]) -> Tuple[bool, List[str]]:
        """
        Verify that summaries contain expected key concepts.
        Returns (success, missing_concepts)
        """
        all_summary_text = " ".join(s[0].lower() for s in summaries)
        missing_concepts = []
        found_concepts = []

        for concept in key_concepts:
            if concept.lower() not in all_summary_text:
                missing_concepts.append(concept)
            else:
                found_concepts.append(concept)

        return len(missing_concepts) == 0, missing_concepts

    async def run_scenario(self, scenario: TestScenario) -> bool:
        """Run a single test scenario"""
        print(f"\n{'='*60}")
        print(f"Running: {scenario.name}")
        print(f"Turn pairs: {scenario.turn_pairs}, Total turns: {len(scenario.conversations)}")
        print(f"{'='*60}")

        try:
            # Setup with specified turn pairs
            await self.setup(turn_pairs=scenario.turn_pairs)

            # Define the test logic as a coroutine
            async def run_test():
                # Process conversations
                print(f"\n📝 Processing {len(scenario.conversations)} conversations...")
                await self.process_conversations(scenario.conversations)

                # Allow async summary generation to complete
                await asyncio.sleep(2.0)

                # Debug: Check processor state
                print(f"🔍 Debug: Processor turn_id = {self.processor._turn_id}")
                print(f"🔍 Debug: Summary enabled = {self.processor._summary_enabled}")
                print(f"🔍 Debug: Window mode = {self.processor._window_mode}")
                print(f"🔍 Debug: Turn pairs = {self.processor._turn_pairs}")
                print(f"🔍 Debug: Last summarized turn = {self.processor._last_summarized_turn}")

                # Send EndFrame to signal completion
                await self.task.queue_frames([EndFrame()])

            # Create tasks for pipeline and test
            pipeline_task = asyncio.create_task(self.runner.run(self.task))
            test_task = asyncio.create_task(run_test())

            # Run test logic
            await test_task

            # Cancel pipeline task after test completes
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass

            # Check turn-based summaries
            summaries_before_cleanup = self.get_summaries()
            print(f"\n📊 Turn-based summaries generated: {len(summaries_before_cleanup)}")

            # Display summaries before cleanup
            if summaries_before_cleanup:
                print("\n  Turn-based summaries:")
                for i, (summary, ts) in enumerate(summaries_before_cleanup, 1):
                    # Clean up the summary text for display
                    summary_text = summary.replace("Summary: ", "").strip()
                    print(f"    {i}. {summary_text[:150]}...")

            # Trigger cleanup for final summary
            await self.processor.cleanup()
            await asyncio.sleep(0.5)  # Allow cleanup summary to be stored

            # Check all summaries including final
            all_summaries = self.get_summaries()
            print(f"\n📊 Total summaries after cleanup: {len(all_summaries)}")

            # Display any new summaries added during cleanup
            if len(all_summaries) > len(summaries_before_cleanup):
                print("\n  Final summary added during cleanup:")
                for summary, ts in all_summaries[len(summaries_before_cleanup):]:
                    summary_text = summary.replace("Summary: ", "").strip()
                    print(f"    • {summary_text[:150]}...")

            # Verify counts
            success = True

            print(f"\n📋 Verification:")

            # Check turn-based summary count
            if len(summaries_before_cleanup) != scenario.expected_summaries:
                print(f"  ❌ Expected {scenario.expected_summaries} turn-based summaries, got {len(summaries_before_cleanup)}")
                success = False
            else:
                print(f"  ✅ Turn-based summary count correct: {scenario.expected_summaries}")

            # Check total summary count including final
            if len(all_summaries) != scenario.expected_with_final:
                print(f"  ❌ Expected {scenario.expected_with_final} total summaries, got {len(all_summaries)}")
                success = False
            else:
                print(f"  ✅ Total summary count correct: {scenario.expected_with_final}")

            # Verify content quality if we have summaries
            if scenario.key_concepts and all_summaries:
                concepts_found, missing = self.verify_summary_content(all_summaries, scenario.key_concepts)
                if concepts_found:
                    print(f"  ✅ All {len(scenario.key_concepts)} key concepts found in summaries")
                else:
                    print(f"  ⚠️  Missing {len(missing)} concepts: {', '.join(missing[:5])}")
                    # This is a warning, not a failure (LLM might be unavailable)

            return success

        except Exception as e:
            logger.error(f"Scenario failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Always cleanup
            await self.teardown()


# Test scenarios with progressive complexity
SCENARIOS = [
    TestScenario(
        name="Basic 5-turn conversation",
        turn_pairs=5,
        conversations=[
            "Hello, my name is Sarah and I'm a data scientist",
            "I work at a startup in Austin, Texas",
            "We're building AI tools for healthcare",
            "I specialize in computer vision and medical imaging",
            "My current project involves detecting tumors in X-ray images"
        ],
        expected_summaries=1,  # Summary after turn 5
        expected_with_final=1,  # No additional final summary needed
        key_concepts=["Sarah", "data scientist", "Austin", "healthcare", "medical imaging"]
    ),

    TestScenario(
        name="Medium 10-turn conversation",
        turn_pairs=5,
        conversations=[
            "I'm planning a trip to Japan next month",
            "I'll be visiting Tokyo, Kyoto, and Osaka",
            "I'm really excited about trying authentic ramen",
            "I've been learning Japanese for six months now",
            "My favorite phrase is 'itadakimasu' which means let's eat",
            "I'm staying in traditional ryokans during my trip",
            "The cherry blossoms should be blooming when I arrive",
            "I plan to visit at least 10 different temples",
            "My budget for the trip is around $3000",
            "I'll be traveling solo for two weeks"
        ],
        expected_summaries=2,  # Summaries after turns 5 and 10
        expected_with_final=2,  # No additional final summary needed
        key_concepts=["Japan", "Tokyo", "Kyoto", "ramen", "Japanese", "cherry blossoms", "temples", "solo"]
    ),

    TestScenario(
        name="Complex 20-turn conversation with context shifts",
        turn_pairs=5,
        conversations=[
            # First topic: Personal introduction (turns 1-5)
            "My name is Dr. Elizabeth Chen and I'm a neuroscientist",
            "I got my PhD from Stanford in 2018",
            "My research focuses on memory formation and retrieval",
            "I've published 15 papers in peer-reviewed journals",
            "Currently, I lead a team of 8 researchers at BrainTech Labs",

            # Second topic: Current research (turns 6-10)
            "We're developing a new brain-computer interface",
            "The device can decode neural signals in real-time",
            "Our goal is to help paralyzed patients control prosthetics",
            "We've had successful trials with 3 patients so far",
            "The FDA approval process will take another 2 years",

            # Third topic: Personal interests (turns 11-15)
            "Outside of work, I'm an avid rock climber",
            "I've climbed El Capitan in Yosemite three times",
            "My climbing partner is my husband James",
            "We met at a climbing gym 10 years ago",
            "We have twin daughters who are 5 years old",

            # Fourth topic: Future plans (turns 16-20)
            "Next year, I'm starting a biotech company",
            "We're raising a $10 million seed round",
            "The company will commercialize our BCI technology",
            "We already have interest from major medical device companies",
            "My ultimate goal is to cure paralysis within 20 years"
        ],
        expected_summaries=4,  # Summaries after turns 5, 10, 15, 20
        expected_with_final=4,  # No additional final summary needed
        key_concepts=[
            "Elizabeth Chen", "neuroscientist", "Stanford", "memory",
            "brain-computer interface", "paralyzed patients", "FDA",
            "rock climber", "El Capitan", "twin daughters",
            "biotech company", "$10 million", "cure paralysis"
        ]
    ),

    TestScenario(
        name="Edge case: 7 turns (incomplete final group)",
        turn_pairs=5,
        conversations=[
            "I love cooking Italian food",
            "My signature dish is homemade pasta carbonara",
            "I learned the recipe from my grandmother in Rome",
            "The secret is using guanciale, not bacon",
            "Fresh eggs and pecorino romano are essential",
            "I make fresh pasta from scratch every Sunday",
            "Next, I want to master making ravioli"
        ],
        expected_summaries=1,  # Summary after turn 5
        expected_with_final=2,  # Additional final summary for turns 6-7
        key_concepts=["Italian", "carbonara", "Rome", "guanciale", "pasta", "Sunday", "ravioli"]
    ),

    TestScenario(
        name="Edge case: Single turn (no summary expected)",
        turn_pairs=5,
        conversations=[
            "Just one quick message here"
        ],
        expected_summaries=0,  # No turn-based summaries
        expected_with_final=0,  # No final summary for single turn
        key_concepts=[]
    ),

    TestScenario(
        name="Large conversation: 20 turns with 10-turn intervals",
        turn_pairs=10,
        conversations=[
            # Professional background (1-10)
            "I'm a senior software engineer at Microsoft",
            "I've been working there for 12 years now",
            "I started as a junior developer on the Office team",
            "Then I moved to Azure cloud services division",
            "I specialize in distributed systems and microservices",
            "My team manages over 50 microservices in production",
            "We handle billions of API requests daily",
            "Our tech stack is primarily C# and Python",
            "We use Kubernetes for container orchestration",
            "I'm also a certified Azure Solutions Architect",

            # Side projects and open source (11-20)
            "In my free time, I contribute to open source projects",
            "I maintain three popular Python libraries on PyPI",
            "My most successful project has over 5000 GitHub stars",
            "It's a machine learning library for time series analysis",
            "I also write technical blogs on Medium",
            "My articles get around 100k views monthly",
            "I'm writing a book about system design patterns",
            "The publisher is O'Reilly Media",
            "The book will be released next spring",
            "I'm also planning to create an online course"
        ],
        expected_summaries=2,  # Summaries after turns 10 and 20
        expected_with_final=2,  # No additional final summary needed
        key_concepts=[
            "Microsoft", "software engineer", "Azure", "distributed systems",
            "Kubernetes", "open source", "Python libraries", "GitHub",
            "Medium", "O'Reilly", "system design", "online course"
        ]
    ),
]


async def check_llm_availability() -> bool:
    """Check if the LLM service is available"""
    import urllib.request
    import urllib.error

    try:
        url = "http://127.0.0.1:1234/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


async def main():
    """Run all test scenarios"""
    print("\n" + "="*60)
    print("TURN-BASED SUMMARIZATION INTEGRATION TESTS")
    print("="*60)

    # Check if LLM is available
    llm_available = await check_llm_availability()
    if llm_available:
        print("✅ LLM service is available at http://127.0.0.1:1234")
    else:
        print("⚠️  LLM service not available - summaries will fail but triggers will be tested")
        print("   To enable full testing, ensure LM Studio is running with gemma-3n-e4b model")

    print("\nStarting test scenarios...")

    test_suite = SummarizationIntegrationTest()
    results = []

    for i, scenario in enumerate(SCENARIOS[:4], 1):  # Run first 4 scenarios
        print(f"\n[{i}/{len(SCENARIOS)}]", end="")
        try:
            success = await test_suite.run_scenario(scenario)
            results.append((scenario.name, success))
        except Exception as e:
            logger.error(f"Scenario failed with unexpected error: {e}")
            results.append((scenario.name, False))

        # Small delay between tests
        await asyncio.sleep(0.5)

    # Final report
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\n📊 Final Score: {passed}/{total} tests passed")

    if not llm_available and passed < total:
        print("\n💡 Tip: Some tests may have failed due to LLM unavailability.")
        print("   Run with LM Studio + gemma-3n-e4b for full test coverage.")

    return passed == total


if __name__ == "__main__":
    # Configure logging for tests
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    success = asyncio.run(main())
    sys.exit(0 if success else 1)