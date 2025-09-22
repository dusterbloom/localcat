"""
Context Formatter - Focused component for formatting memory context.

This component handles the formatting of memory bullets for injection
into LLM context. It follows the Single Responsibility Principle by
focusing solely on context formatting and bullet generation.
"""

from typing import List, Dict, Any, Optional
from loguru import logger


class ContextFormatter:
    """
    Focused component for formatting memory bullets into LLM context.
    Handles bullet formatting, truncation, and context injection.
    """

    def __init__(self,
                 max_bullets: int = 3,
                 inject_role: str = "system",
                 inject_header: str = "Use the following factual context if helpful."):
        self.max_bullets = max_bullets
        self.inject_role = inject_role
        self.inject_header = inject_header

    def format_bullets(self, bullets: List[str], max_bullets: Optional[int] = None) -> List[str]:
        """
        Format raw memory bullets for context injection.

        Args:
            bullets: Raw bullet strings
            max_bullets: Override default max bullets

        Returns:
            Formatted bullet strings
        """
        if not bullets:
            return []

        max_count = max_bullets if max_bullets is not None else self.max_bullets
        capped_bullets = bullets[:max_count]

        # Clean and format bullets
        formatted = []
        for bullet in capped_bullets:
            cleaned = self._clean_bullet(bullet)
            if cleaned:
                formatted.append(cleaned)

        return formatted

    def build_message(self, role: str, header: str, bullets: List[str]) -> Dict[str, str]:
        """
        Build a complete context message with header and bullets.

        Args:
            role: Message role (system/user)
            header: Context header text
            bullets: Formatted bullet strings

        Returns:
            Complete message dictionary
        """
        if not bullets:
            return None

        content_parts = [header]

        # Add bullets with proper formatting
        for bullet in bullets:
            content_parts.append(f"• {bullet}")

        content = "\n".join(content_parts)

        return {
            "role": role,
            "content": content
        }

    def _clean_bullet(self, bullet: str) -> str:
        """
        Clean and normalize a memory bullet.

        Args:
            bullet: Raw bullet text

        Returns:
            Cleaned bullet text
        """
        if not bullet or not bullet.strip():
            return ""

        # Remove excessive whitespace
        cleaned = " ".join(bullet.split())

        # Remove any leading/trailing punctuation that might interfere
        cleaned = cleaned.strip(".,;:!? ")

        # Basic length check
        if len(cleaned) < 3:
            return ""

        return cleaned

    def truncate_bullets(self, bullets: List[str], max_length: int = 500) -> List[str]:
        """
        Truncate bullets to fit within maximum context length.

        Args:
            bullets: Formatted bullets
            max_length: Maximum total length

        Returns:
            Truncated bullet list
        """
        if not bullets:
            return []

        result = []
        current_length = 0

        for bullet in bullets:
            bullet_length = len(bullet) + 2  # +2 for "• " prefix

            if current_length + bullet_length > max_length:
                break

            result.append(bullet)
            current_length += bullet_length

        return result

    def get_injection_config(self) -> Dict[str, Any]:
        """
        Get current injection configuration.

        Returns:
            Configuration dictionary
        """
        return {
            "max_bullets": self.max_bullets,
            "inject_role": self.inject_role,
            "inject_header": self.inject_header
        }

    def update_config(self, **kwargs) -> None:
        """
        Update formatter configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")