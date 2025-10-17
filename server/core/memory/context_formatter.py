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
        Deduplicates while preserving order and source tags.

        Args:
            bullets: Raw bullet strings (may include [convo]/[graph]/[summary] tags)
            max_bullets: Override default max bullets

        Returns:
            Formatted bullet strings with source tags preserved
        """
        if not bullets:
            return []

        max_count = max_bullets if max_bullets is not None else self.max_bullets

        # Deduplicate while preserving order
        seen = set()
        unique_bullets = []
        for bullet in bullets:
            if bullet not in seen:
                seen.add(bullet)
                unique_bullets.append(bullet)

        # Cap to max count
        capped_bullets = unique_bullets[:max_count]

        # Clean and format bullets (preserving source tags)
        formatted = []
        for bullet in capped_bullets:
            cleaned = self._clean_bullet_preserve_tags(bullet)
            if cleaned:
                formatted.append(cleaned)

        return formatted

    def build_message(self, role: str, header: str, bullets: List[str]) -> Dict[str, str]:
        """
        Build a complete context message with header and bullets.

        Args:
            role: Message role (system/user)
            header: Context header text
            bullets: Formatted bullet strings (already include "• " prefix from retrieval)

        Returns:
            Complete message dictionary
        """
        if not bullets:
            return None

        # Bullets already have "• " prefix from retrieval, just join with header
        content = header.rstrip() + "\n" + "\n".join(bullets)

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

    def _clean_bullet_preserve_tags(self, bullet: str) -> str:
        """
        Clean bullet while preserving [source] tags and timestamps.
        Bullets from retrieval already have format: "• [convo] text... (timestamp)"

        Args:
            bullet: Raw bullet text with source tags

        Returns:
            Cleaned bullet text with tags preserved
        """
        if not bullet or not bullet.strip():
            return ""

        # Bullets already have "• " prefix from retrieval - keep as-is
        cleaned = bullet.strip()

        # Normalize internal whitespace (but preserve structure)
        import re
        # Replace multiple spaces with single space, but keep newlines
        cleaned = re.sub(r' +', ' ', cleaned)

        # Basic quality check - ensure there's actual content beyond the prefix and tag
        # Pattern: "• [source] actual_content (timestamp)"
        content_only = re.sub(r'^•\s*\[.*?\]\s*', '', cleaned)
        content_only = re.sub(r'\s*\(.*?\)\s*$', '', content_only)

        if len(content_only.strip()) < 5:
            # Reject bullets with insufficient content
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