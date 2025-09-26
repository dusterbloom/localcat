"""
Intent Classification Module for LocalCat Voice Agent

This module provides intent classification services to enable smart routing
of user utterances for optimized memory processing and conversation flow.
"""

from .intent_service import IntentService, get_intent_service

__all__ = ['IntentService', 'get_intent_service']