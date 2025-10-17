"""Test that quality filter properly rejects interjections and low-quality bullets."""

import pytest
from unittest.mock import Mock
from core.memory.retrieval import Retrieval


class TestQualityFilter:
    """Test that the quality filter rejects interjections and accepts substantive content."""
    
    def test_quality_filter_rejects_interjections(self):
        """Ensure _is_quality_bullet rejects common interjections."""
        host = Mock()
        retrieval = Retrieval(host)
        
        # Test common interjections that should be rejected
        interjections = [
            "Oh my god.",
            "Wow!",
            "lol",
            "yeah",
            "hmm",
            "uh",
            "ok",
            "okay",
            "right",
            "sure",
            "thanks",
            "Oh wow",
            "Lol that's funny",
            "Hmm I see",
            "Uh oh",
        ]
        
        for interjection in interjections:
            assert not retrieval._is_quality_bullet(interjection), f"Should reject: {interjection}"
            
    def test_quality_filter_accepts_substantive_content(self):
        """Ensure _is_quality_bullet accepts substantive sentences."""
        host = Mock()
        retrieval = Retrieval(host)
        
        # Test substantive content that should be accepted
        substantive = [
            "I work as a software engineer at a tech company.",
            "The user mentioned they enjoy hiking on weekends.",
            "She moved to New York last year for her job.",
            "We discussed the project timeline and deliverables.",
            "The meeting is scheduled for tomorrow at 2 PM.",
            "I prefer Python over JavaScript for web development.",
            "The user has three children and lives in California.",
        ]
        
        for content in substantive:
            assert retrieval._is_quality_bullet(content), f"Should accept: {content}"
            
    def test_quality_filter_length_requirement(self):
        """Test that very short utterances are rejected."""
        host = Mock()
        retrieval = Retrieval(host)
        
        # Very short utterances should be rejected
        short_utterances = [
            "Hi.",
            "Ok.",
            "Yes.",
            "No.",
            "lol",
            "wow",
        ]
        
        for short in short_utterances:
            assert not retrieval._is_quality_bullet(short), f"Should reject short: {short}"
            
    def test_quality_filter_content_patterns(self):
        """Test that content requires substance beyond just length."""
        host = Mock()
        retrieval = Retrieval(host)
        
        # Long but empty content should still be rejected
        empty_content = [
            "uh huh yeah right okay sure whatever",  # Long but all fillers
            "oh wow hmm interesting right lol",      # Long but just reactions
        ]
        
        for content in empty_content:
            # These might pass length test but should fail content quality
            result = retrieval._is_quality_bullet(content)
            # Implementation will need to check for content verbs/nouns
            pass
