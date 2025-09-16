"""
Context budget management for token allocation in progressive context system.
"""
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, NamedTuple
from functools import lru_cache

from .exceptions import BudgetError, ValidationError

logger = logging.getLogger(__name__)


class BudgetAllocations(NamedTuple):
    """Named tuple for type-safe budget allocations"""
    system: int
    memory: int
    summary: int
    dialogue: int
    total: int


@dataclass
class ContextBudget:
    """Manages token budget allocation for context packing with progressive disclosure"""

    # Budget ratios - these determine how total budget is distributed
    system_ratio: float = 0.12    # 12% for system instructions
    memory_ratio: float = 0.25    # 25% for memory context (increased from original 15%)
    summary_ratio: float = 0.10   # 10% for session summaries
    # dialogue gets the remainder (~53%)

    # Absolute maximums to prevent any section from growing too large
    max_system_tokens: int = 512
    max_memory_tokens: int = 1200   # Matches original allocation
    max_summary_tokens: int = 400

    # Total budget
    total_budget: int = 4096

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()

    @classmethod
    @lru_cache(maxsize=1)
    def from_env(cls) -> 'ContextBudget':
        """
        Load budget configuration from environment variables.

        Environment variables:
        - CONTEXT_BUDGET_TOKENS: Total token budget (default: 4096)
        - CONTEXT_SYSTEM_RATIO: Ratio for system instructions (default: 0.12)
        - CONTEXT_MEMORY_RATIO: Ratio for memory context (default: 0.25)
        - CONTEXT_SUMMARY_RATIO: Ratio for summaries (default: 0.10)
        - CONTEXT_MAX_SYSTEM: Max tokens for system (default: 512)
        - CONTEXT_MAX_MEMORY: Max tokens for memory (default: 1200)
        - CONTEXT_MAX_SUMMARY: Max tokens for summary (default: 400)

        Returns:
            ContextBudget configured from environment
        """
        try:
            return cls(
                system_ratio=float(os.getenv("CONTEXT_SYSTEM_RATIO", "0.12")),
                memory_ratio=float(os.getenv("CONTEXT_MEMORY_RATIO", "0.25")),
                summary_ratio=float(os.getenv("CONTEXT_SUMMARY_RATIO", "0.10")),
                max_system_tokens=int(os.getenv("CONTEXT_MAX_SYSTEM", "512")),
                max_memory_tokens=int(os.getenv("CONTEXT_MAX_MEMORY", "1200")),
                max_summary_tokens=int(os.getenv("CONTEXT_MAX_SUMMARY", "400")),
                total_budget=int(os.getenv("CONTEXT_BUDGET_TOKENS", "4096"))
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid environment configuration for ContextBudget: {e}")
            logger.info("Using default ContextBudget configuration")
            return cls()  # Use defaults

    def get_allocations(self) -> BudgetAllocations:
        """
        Calculate token allocations for each context section.

        Returns:
            BudgetAllocations with calculated token limits for each section
        """
        # Calculate base allocations from ratios
        system_base = int(self.total_budget * self.system_ratio)
        memory_base = int(self.total_budget * self.memory_ratio)
        summary_base = int(self.total_budget * self.summary_ratio)

        # Apply maximum limits
        system_tokens = min(system_base, self.max_system_tokens)
        memory_tokens = min(memory_base, self.max_memory_tokens)
        summary_tokens = min(summary_base, self.max_summary_tokens)

        # Calculate dialogue allocation (remainder)
        reserved_tokens = system_tokens + memory_tokens + summary_tokens
        dialogue_tokens = max(0, self.total_budget - reserved_tokens)

        return BudgetAllocations(
            system=system_tokens,
            memory=memory_tokens,
            summary=summary_tokens,
            dialogue=dialogue_tokens,
            total=self.total_budget
        )

    def validate(self) -> None:
        """
        Validate budget configuration for logical consistency.

        Raises:
            BudgetError: If configuration is invalid
        """
        errors = []

        # Check that ratios are reasonable
        total_ratio = self.system_ratio + self.memory_ratio + self.summary_ratio
        if total_ratio >= 1.0:
            errors.append(f"Budget ratios sum to {total_ratio:.2f}, must be < 1.0 to leave room for dialogue")

        # Check that ratios are positive
        if any(ratio < 0 for ratio in [self.system_ratio, self.memory_ratio, self.summary_ratio]):
            errors.append("All budget ratios must be non-negative")

        # Check for unreasonably small ratios
        if self.system_ratio < 0.05:
            logger.warning(f"System ratio ({self.system_ratio}) is very small, may cause issues")
        if self.memory_ratio < 0.05:
            logger.warning(f"Memory ratio ({self.memory_ratio}) is very small, may cause issues")

        # Check that maximums are reasonable
        if any(max_tokens <= 0 for max_tokens in [
            self.max_system_tokens, self.max_memory_tokens, self.max_summary_tokens
        ]):
            errors.append("All maximum token limits must be positive")

        # Check that total budget is reasonable
        if self.total_budget <= 0:
            errors.append("Total budget must be positive")
        elif self.total_budget < 512:
            logger.warning(f"Total budget ({self.total_budget}) is very small, may cause issues")

        # Check for internally consistent maximums
        if self.max_system_tokens > self.total_budget:
            logger.warning(f"max_system_tokens ({self.max_system_tokens}) exceeds total budget")
        if self.max_memory_tokens > self.total_budget:
            logger.warning(f"max_memory_tokens ({self.max_memory_tokens}) exceeds total budget")

        if errors:
            raise BudgetError(f"Budget validation failed: {'; '.join(errors)}")

        # Warn if maximums are too restrictive (but don't error)
        try:
            allocations = self.get_allocations()
            min_dialogue = allocations.dialogue

            if min_dialogue < self.total_budget * 0.3:  # Less than 30% for dialogue
                logger.warning(
                    f"Dialogue allocation ({min_dialogue} tokens) is less than 30% of total budget. "
                    "Consider reducing other allocations to ensure sufficient dialogue space."
                )
        except Exception as e:
            logger.warning(f"Could not validate dialogue allocation: {e}")

    def get_usage_info(self, actual_usage: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate usage percentages for monitoring and optimization.

        Args:
            actual_usage: Dictionary with keys: system, memory, summary, dialogue

        Returns:
            Dictionary with usage percentages for each section
        """
        allocations = self.get_allocations()

        usage_info = {}

        for section in ['system', 'memory', 'summary', 'dialogue']:
            allocated = getattr(allocations, section)
            used = actual_usage.get(section, 0)

            if allocated > 0:
                usage_pct = (used / allocated) * 100
                usage_info[f"{section}_usage_pct"] = round(usage_pct, 1)
                usage_info[f"{section}_allocated"] = allocated
                usage_info[f"{section}_used"] = used
            else:
                usage_info[f"{section}_usage_pct"] = 0.0
                usage_info[f"{section}_allocated"] = 0
                usage_info[f"{section}_used"] = used

        total_used = sum(actual_usage.values())
        usage_info["total_usage_pct"] = round((total_used / self.total_budget) * 100, 1)
        usage_info["total_allocated"] = self.total_budget
        usage_info["total_used"] = total_used

        return usage_info

    def suggest_optimizations(self, usage_stats: Dict[str, int]) -> List[str]:
        """
        Suggest budget optimizations based on actual usage patterns.

        Args:
            usage_stats: Dictionary with actual token usage per section

        Returns:
            List of optimization suggestions
        """
        suggestions = []
        allocations = self.get_allocations()
        usage_info = self.get_usage_info(usage_stats)

        # Check for consistently underutilized sections
        if usage_info.get("system_usage_pct", 0) < 50:
            suggestions.append(
                f"System section using only {usage_info.get('system_usage_pct', 0):.1f}% "
                f"of allocation. Consider reducing CONTEXT_SYSTEM_RATIO from {self.system_ratio}"
            )

        if usage_info.get("memory_usage_pct", 0) < 50:
            suggestions.append(
                f"Memory section using only {usage_info.get('memory_usage_pct', 0):.1f}% "
                f"of allocation. Consider reducing CONTEXT_MEMORY_RATIO from {self.memory_ratio}"
            )

        if usage_info.get("summary_usage_pct", 0) < 50:
            suggestions.append(
                f"Summary section using only {usage_info.get('summary_usage_pct', 0):.1f}% "
                f"of allocation. Consider reducing CONTEXT_SUMMARY_RATIO from {self.summary_ratio}"
            )

        # Check for consistently over-utilized sections
        if usage_info.get("dialogue_usage_pct", 0) > 90:
            suggestions.append(
                "Dialogue section consistently near capacity. Consider increasing total budget "
                "or reducing allocations for other sections."
            )

        return suggestions


# Global instance for convenience (lazy-loaded)
_global_budget = None

def get_global_budget() -> ContextBudget:
    """Get a global ContextBudget instance (singleton pattern)"""
    global _global_budget
    if _global_budget is None:
        _global_budget = ContextBudget.from_env()
    return _global_budget