"""
Unit tests for configuration parsing utilities.
"""

import pytest
from config.parsers import (
    _parse_bool,
    _parse_int,
    _parse_float,
    _parse_list,
    _parse_enum,
    parse_with_validator,
)


class TestParseBool:
    """Test boolean parsing."""

    def test_parse_true_values(self):
        """Test various true value formats."""
        assert _parse_bool("true") is True
        assert _parse_bool("TRUE") is True
        assert _parse_bool("True") is True
        assert _parse_bool("1") is True
        assert _parse_bool("yes") is True
        assert _parse_bool("YES") is True
        assert _parse_bool("on") is True
        assert _parse_bool("ON") is True

    def test_parse_false_values(self):
        """Test various false value formats."""
        assert _parse_bool("false") is False
        assert _parse_bool("FALSE") is False
        assert _parse_bool("0") is False
        assert _parse_bool("no") is False
        assert _parse_bool("off") is False
        assert _parse_bool("") is False
        assert _parse_bool("invalid") is False

    def test_parse_none(self):
        """Test None input."""
        assert _parse_bool(None) is False


class TestParseInt:
    """Test integer parsing."""

    def test_parse_valid_int(self):
        """Test valid integer values."""
        assert _parse_int("42", 0) == 42
        assert _parse_int("0", 10) == 0
        assert _parse_int("-5", 0) == -5
        assert _parse_int("1000000", 0) == 1000000

    def test_parse_invalid_int(self):
        """Test invalid integer values fall back to default."""
        assert _parse_int("invalid", 10) == 10
        assert _parse_int("3.14", 5) == 5
        assert _parse_int("", 7) == 7
        assert _parse_int("abc123", 0) == 0

    def test_parse_none(self):
        """Test None input returns default."""
        assert _parse_int(None, 42) == 42


class TestParseFloat:
    """Test float parsing."""

    def test_parse_valid_float(self):
        """Test valid float values."""
        assert _parse_float("3.14", 0.0) == 3.14
        assert _parse_float("0.0", 1.0) == 0.0
        assert _parse_float("-2.5", 0.0) == -2.5
        assert _parse_float("42", 0.0) == 42.0
        assert _parse_float("1e-3", 0.0) == 0.001

    def test_parse_invalid_float(self):
        """Test invalid float values fall back to default."""
        assert _parse_float("invalid", 1.5) == 1.5
        assert _parse_float("", 2.0) == 2.0
        assert _parse_float("abc", 0.5) == 0.5

    def test_parse_none(self):
        """Test None input returns default."""
        assert _parse_float(None, 3.14) == 3.14


class TestParseList:
    """Test list parsing."""

    def test_parse_valid_list(self):
        """Test valid comma-separated lists."""
        assert _parse_list("a,b,c", []) == ["a", "b", "c"]
        assert _parse_list("foo", []) == ["foo"]
        assert _parse_list("one,two,three,four", []) == ["one", "two", "three", "four"]

    def test_parse_with_whitespace(self):
        """Test list parsing with whitespace."""
        assert _parse_list("  a  ,  b  ,  c  ", []) == ["a", "b", "c"]
        assert _parse_list(" foo , bar ", []) == ["foo", "bar"]

    def test_parse_empty_values(self):
        """Test list parsing filters empty values."""
        assert _parse_list("a,,b", []) == ["a", "b"]
        assert _parse_list(",,,", []) == []
        assert _parse_list("", []) == []

    def test_parse_custom_separator(self):
        """Test custom separator."""
        assert _parse_list("a;b;c", [], separator=";") == ["a", "b", "c"]
        assert _parse_list("a|b|c", [], separator="|") == ["a", "b", "c"]

    def test_parse_none(self):
        """Test None input returns default."""
        assert _parse_list(None, ["default"]) == ["default"]
        assert _parse_list(None, []) == []


class TestParseEnum:
    """Test enum/choice parsing."""

    def test_parse_valid_enum(self):
        """Test valid enum values."""
        assert _parse_enum("debug", "info", ["debug", "info", "warning"]) == "debug"
        assert _parse_enum("info", "debug", ["debug", "info", "warning"]) == "info"

    def test_parse_invalid_enum(self):
        """Test invalid enum values fall back to default."""
        assert _parse_enum("invalid", "info", ["debug", "info"]) == "info"
        assert _parse_enum("trace", "debug", ["debug", "info"]) == "debug"

    def test_parse_none(self):
        """Test None input returns default."""
        assert _parse_enum(None, "info", ["debug", "info"]) == "info"


class TestParseWithValidator:
    """Test custom parser with validation."""

    def test_parse_with_valid_value(self):
        """Test valid value passes validation."""
        result = parse_with_validator("10", int, 5, lambda x: x > 0, "Must be positive")
        assert result == 10

    def test_parse_with_invalid_value(self):
        """Test invalid value fails validation."""
        result = parse_with_validator("-5", int, 5, lambda x: x > 0, "Must be positive")
        assert result == 5

    def test_parse_with_no_validator(self):
        """Test parsing without validator."""
        result = parse_with_validator("10", int, 5)
        assert result == 10

    def test_parse_with_parsing_error(self):
        """Test parsing error returns default."""
        result = parse_with_validator("invalid", int, 5, lambda x: x > 0)
        assert result == 5

    def test_parse_none_returns_default(self):
        """Test None input returns default."""
        result = parse_with_validator(None, int, 5, lambda x: x > 0)
        assert result == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
