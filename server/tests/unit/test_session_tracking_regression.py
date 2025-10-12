#!/usr/bin/env python3
"""
Unit tests for session tracking regression fix.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add server to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.hotpath_processor import HotPathMemoryProcessor

class TestSessionTrackingRegression(unittest.TestCase):
    """Test session tracking regression fixes."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock dependencies
        self.mock_context = Mock()
        self.mock_context.get_messages.return_value = []  # Return empty list for initialization
        
        self.mock_context_aggregator = Mock()
        self.mock_context_aggregator.user.return_value = self.mock_context
        
        self.mock_session_tracker = Mock()
        self.mock_session_tracker.start_session.return_value = {
            "total_sessions": 5,
            "current_session": 3,
            "session_turns": 0,
            "session_start_iso": "2025-10-12 19:00:00",
            "session_elapsed": 0,
            "total_time_seconds": 3600
        }
        
    def test_user_id_normalization_for_session_tracking(self):
        """Test that user_id normalization prevents case sensitivity issues."""
        # Test various case combinations
        test_cases = [
            ("peppi", "peppi"),  # Same case as env
            ("Peppi", "peppi"),  # Different case, should normalize
            ("PEPPI", "peppi"),  # All caps, should normalize
            ("PePpI", "peppi"),  # Mixed case, should normalize
        ]
        
        for input_user, expected in test_cases:
            with patch.dict(os.environ, {"USER_ID": "peppi"}):
                # Import the method directly to test it
                from core.memory.hotpath_processor import HotPathMemoryProcessor
                
                # Create a minimal instance just to test the method
                processor = object.__new__(HotPathMemoryProcessor)
                
                result = processor._normalize_user_id_for_session(input_user)
                self.assertEqual(result, expected, f"Failed to normalize '{input_user}'")
    
    def test_set_user_identity_preserves_session_tracking(self):
        """Test that set_user_identity preserves session tracking compatibility."""
        processor = HotPathMemoryProcessor(
            user_id="peppi",
            session_tracker=self.mock_session_tracker,
            context_aggregator=self.mock_context_aggregator
        )
        
        # Simulate speaker recognition calling set_user_identity with capitalized name
        processor.set_user_identity("Peppi")
        
        # Verify that session tracking uses normalized user_id
        self.assertEqual(processor._user_id, "peppi")  # Should be normalized for sessions
        self.assertEqual(processor._display_user_id, "Peppi")  # Should preserve original for display
        
        # Verify that session tracker would be called with normalized user_id
        self.mock_session_tracker.start_session.assert_called()
        call_args = self.mock_session_tracker.start_session.call_args[0]
        self.assertEqual(call_args[0], "peppi")  # First argument should be normalized user_id
    
    def test_session_header_uses_display_user_id(self):
        """Test that session headers use the display user ID with proper capitalization."""
        processor = HotPathMemoryProcessor(
            user_id="peppi",
            session_tracker=self.mock_session_tracker,
            context_aggregator=self.mock_context_aggregator
        )
        
        # Set user identity with capitalization
        processor.set_user_identity("Peppi")
        
        # Mock stats
        stats = {
            "total_sessions": 10,
            "current_session": 5,
            "session_turns": 3,
            "session_start_iso": "2025-10-12 19:00:00",
            "session_elapsed": 300,
            "total_time_seconds": 7200
        }
        
        # Build session header
        header = processor._build_session_header(stats)
        
        # Verify header contains the properly capitalized user name
        header_content = header["content"]
        self.assertIn("User: Peppi", header_content)
        self.assertIn("Session #5", header_content)
        self.assertIn("Total sessions: 10", header_content)
    
    def test_anonymous_mode_display(self):
        """Test that anonymous mode displays correctly."""
        processor = HotPathMemoryProcessor(
            user_id="peppi",
            session_tracker=self.mock_session_tracker,
            context_aggregator=self.mock_context_aggregator
        )
        
        # Enable anonymous mode
        processor.set_ephemeral_mode(True)
        
        # Mock stats
        stats = {
            "total_sessions": 1,
            "current_session": 1,
            "session_turns": 1,
            "session_start_iso": "2025-10-12 19:00:00",
            "session_elapsed": 60,
            "total_time_seconds": 60
        }
        
        # Build session header
        header = processor._build_session_header(stats)
        
        # Verify header shows anonymous
        header_content = header["content"]
        self.assertIn("User: anonymous", header_content)

if __name__ == "__main__":
    unittest.main()
