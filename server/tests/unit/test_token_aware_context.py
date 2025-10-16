"""
Test token-aware context pruning to prevent performance degradation.
"""

import pytest
from unittest.mock import Mock
from core.memory.token_estimator import TokenEstimator
from core.memory.context_injector import ContextInjector
from core.memory.config_manager import MemoryConfiguration


def test_token_estimator_basic():
    """Test basic token estimation."""
    # Test simple text
    text = "Hello, how are you?"
    tokens = TokenEstimator.estimate_tokens(text)
    assert tokens > 0
    assert isinstance(tokens, int)

    # Test empty text
    assert TokenEstimator.estimate_tokens("") == 0

    # Test message estimation
    message = {"role": "user", "content": "This is a test message"}
    msg_tokens = TokenEstimator.estimate_message_tokens(message)
    assert msg_tokens > TokenEstimator.estimate_tokens("This is a test message")  # Should include overhead


def test_token_estimator_messages():
    """Test token estimation for message lists."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"}
    ]

    total_tokens = TokenEstimator.estimate_messages_tokens(messages)
    assert total_tokens > 0
    assert isinstance(total_tokens, int)

    # Should be more than just the content tokens (includes overhead)
    content_tokens = sum(
        TokenEstimator.estimate_tokens(msg["content"])
        for msg in messages
    )
    assert total_tokens > content_tokens


def test_context_pruning_token_aware():
    """Test token-aware context pruning."""
    # Create configuration with low token budget to force pruning
    config = MemoryConfiguration()
    config.llm_context_max_tokens = 500  # Small budget to force pruning
    config.llm_context_prune_threshold = 0.70
    config.llm_context_min_turns = 2
    config.ctx_window_enabled = True

    # Create mock hot memory and context aggregator
    mock_hot = Mock()
    mock_context_agg = Mock()

    # Create context injector
    injector = ContextInjector(
        hot_memory=mock_hot,
        config=config,
        context_aggregator=mock_context_agg
    )

    # Create a long conversation that exceeds token budget
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": "[Session Context] User: test-user, Session: abc123"},
        {"role": "system", "content": "Use the following factual context if helpful."},
    ]

    # Add many user/assistant turns
    for i in range(20):
        messages.append({"role": "user", "content": f"This is user message {i} with some content to consume tokens."})
        messages.append({"role": "assistant", "content": f"This is assistant response {i} with detailed information that takes up space."})

    # Prune the context
    pruned = injector._prune_context_window(messages)

    # Verify system messages are kept
    system_msgs = [m for m in pruned if m.get("role") == "system"]
    assert len(system_msgs) == 3  # All system messages should be kept

    # Verify some conversation history was pruned
    ua_msgs = [m for m in pruned if m.get("role") in ("user", "assistant")]
    assert len(ua_msgs) < 40  # Should have pruned some of the 40 turn pairs
    assert len(ua_msgs) >= 4  # Should keep at least min_turns * 2 (2 turns = 4 messages)

    # Verify total token count is within budget
    total_tokens = TokenEstimator.estimate_messages_tokens(pruned)
    budget = int(config.llm_context_max_tokens * config.llm_context_prune_threshold)
    assert total_tokens <= budget, f"Pruned context {total_tokens} exceeds budget {budget}"


def test_context_pruning_keeps_minimum_turns():
    """Test that pruning always keeps minimum turns even if over budget."""
    config = MemoryConfiguration()
    config.llm_context_max_tokens = 100  # Very small budget
    config.llm_context_prune_threshold = 0.70
    config.llm_context_min_turns = 3  # Must keep 3 turns minimum
    config.ctx_window_enabled = True

    mock_hot = Mock()
    mock_context_agg = Mock()

    injector = ContextInjector(
        hot_memory=mock_hot,
        config=config,
        context_aggregator=mock_context_agg
    )

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User message 1 with lots of content to exceed the budget"},
        {"role": "assistant", "content": "Assistant response 1 with lots of content to exceed the budget"},
        {"role": "user", "content": "User message 2 with lots of content to exceed the budget"},
        {"role": "assistant", "content": "Assistant response 2 with lots of content to exceed the budget"},
        {"role": "user", "content": "User message 3 with lots of content to exceed the budget"},
        {"role": "assistant", "content": "Assistant response 3 with lots of content to exceed the budget"},
    ]

    pruned = injector._prune_context_window(messages)

    # Should keep system message + minimum 3 turns (6 messages)
    ua_msgs = [m for m in pruned if m.get("role") in ("user", "assistant")]
    assert len(ua_msgs) >= 6  # 3 turns = 6 messages (user + assistant)


def test_context_pruning_keeps_pairs():
    """Test that pruning maintains complete user/assistant pairs."""
    config = MemoryConfiguration()
    config.llm_context_max_tokens = 600
    config.llm_context_prune_threshold = 0.70
    config.llm_context_min_turns = 2
    config.ctx_window_enabled = True

    mock_hot = Mock()
    mock_context_agg = Mock()

    injector = ContextInjector(
        hot_memory=mock_hot,
        config=config,
        context_aggregator=mock_context_agg
    )

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User 1"},
        {"role": "assistant", "content": "Assistant 1"},
        {"role": "user", "content": "User 2"},
        {"role": "assistant", "content": "Assistant 2"},
        {"role": "user", "content": "User 3"},
        {"role": "assistant", "content": "Assistant 3"},
    ]

    pruned = injector._prune_context_window(messages)

    # Count user and assistant messages
    ua_msgs = [m for m in pruned if m.get("role") in ("user", "assistant")]
    user_count = sum(1 for m in ua_msgs if m.get("role") == "user")
    assistant_count = sum(1 for m in ua_msgs if m.get("role") == "assistant")

    # Should have equal number of user and assistant messages (complete pairs)
    assert user_count == assistant_count


def test_context_pruning_fallback_on_error():
    """Test that pruning falls back gracefully on errors."""
    config = MemoryConfiguration()
    config.llm_context_max_tokens = 3000
    config.ctx_window_enabled = True
    config.ctx_max_pairs = 4  # Fallback setting

    mock_hot = Mock()
    mock_context_agg = Mock()

    injector = ContextInjector(
        hot_memory=mock_hot,
        config=config,
        context_aggregator=mock_context_agg
    )

    # Create messages that might cause errors (malformed)
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "User 1"},
        {"role": "assistant", "content": "Assistant 1"},
        {"role": "user", "content": "User 2"},
        {"role": "assistant", "content": "Assistant 2"},
        {"role": "user", "content": "User 3"},
        {"role": "assistant", "content": "Assistant 3"},
        {"role": "user", "content": "User 4"},
        {"role": "assistant", "content": "Assistant 4"},
        {"role": "user", "content": "User 5"},
        {"role": "assistant", "content": "Assistant 5"},
    ]

    # Should not raise exception
    pruned = injector._prune_context_window(messages)
    assert len(pruned) > 0
    assert any(m.get("role") == "system" for m in pruned)


def test_token_estimator_metrics():
    """Test token estimator metrics."""
    metrics = TokenEstimator.get_metrics()

    assert "tiktoken_available" in metrics
    assert "encoding" in metrics
    assert "encoder_loaded" in metrics

    assert isinstance(metrics["tiktoken_available"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
