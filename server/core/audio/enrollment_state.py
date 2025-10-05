"""
Enrollment State Machine - Single Responsibility Principle

Defines the state model for speaker enrollment pipeline transitions.
This module only handles state representation - no business logic.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class EnrollmentState(Enum):
    """Pipeline states for enrollment flow"""
    CHOICE = "choice"            # Offer: enroll vs anonymous
    INTRO = "intro"              # Initial greeting + explanation
    ENROLLING = "enrolling"       # Collecting voice samples (1/3, 2/3, 3/3)
    TRANSITION = "transition"     # "All set!" acknowledgment
    NAME_CAPTURE = "name_capture" # Ask user for preferred ID/name
    CONVERSATION = "conversation" # Main pipeline active
    
    def __str__(self) -> str:
        return self.value


@dataclass
class EnrollmentProgress:
    """
    Progress tracking for enrollment process.
    Immutable data structure following value object pattern.
    """
    current: int
    total: int
    state: EnrollmentState
    speaker_id: Optional[str] = None
    consistency: float = 0.0
    
    @property
    def percentage(self) -> float:
        """Progress as percentage (0-100)"""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if enrollment is complete"""
        return self.current >= self.total
    
    def __str__(self) -> str:
        return f"EnrollmentProgress({self.current}/{self.total}, {self.state.value})"
