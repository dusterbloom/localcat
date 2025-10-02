"""
Enrollment Message Templates - DRY Principle

Centralized message templates for enrollment flow.
Single source of truth for all user-facing enrollment messages.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnrollmentMessages:
    """
    Centralized enrollment messages (DRY - single source of truth).
    
    All messages can be customized via environment variables or constructor.
    """
    
    intro_message: str = (
        "Hi! I'm LocalCat. I can learn to recognize your voice for a more "
        "personalized experience. Just speak naturally for a moment, and I'll "
        "remember you next time!"
    )
    
    progress_template: str = "Learning your voice... {current} of {total}"
    
    completion_template: str = (
        "Perfect! I'll remember you{name}. How can I help you today?"
    )
    
    welcome_back_template: str = "Welcome back{name}! What can I help you with?"
    
    privacy_explanation: str = (
        "I store your voice profile locally on your device. "
        "You can delete it anytime by removing your speaker profile."
    )
    
    @classmethod
    def from_env(cls) -> 'EnrollmentMessages':
        """
        Load messages from environment variables with fallback to defaults.
        Supports customization without code changes.
        """
        return cls(
            intro_message=os.getenv(
                "AUDIO_INTEL_INTRO_MESSAGE",
                cls.intro_message
            ),
            progress_template=os.getenv(
                "AUDIO_INTEL_PROGRESS_TEMPLATE",
                cls.progress_template
            ),
            completion_template=os.getenv(
                "AUDIO_INTEL_COMPLETION_TEMPLATE",
                cls.completion_template
            ),
            welcome_back_template=os.getenv(
                "AUDIO_INTEL_WELCOME_BACK_TEMPLATE",
                cls.welcome_back_template
            ),
            privacy_explanation=os.getenv(
                "AUDIO_INTEL_PRIVACY_EXPLANATION",
                cls.privacy_explanation
            ),
        )
    
    def get_intro(self, include_privacy: bool = False) -> str:
        """Get introduction message with optional privacy explanation"""
        msg = self.intro_message
        if include_privacy:
            msg += f" {self.privacy_explanation}"
        return msg
    
    def get_progress(self, current: int, total: int) -> str:
        """Get progress message (e.g., "Learning your voice... 2 of 3")"""
        return self.progress_template.format(current=current, total=total)
    
    def get_completion(self, speaker_name: Optional[str] = None) -> str:
        """Get completion message with optional speaker name"""
        name = f", {speaker_name}" if speaker_name else ""
        return self.completion_template.format(name=name)
    
    def get_welcome_back(self, speaker_name: Optional[str] = None) -> str:
        """Get welcome back message for returning users"""
        name = f", {speaker_name}" if speaker_name else ""
        return self.welcome_back_template.format(name=name)
