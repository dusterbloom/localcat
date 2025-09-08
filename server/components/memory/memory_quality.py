"""
Memory Quality Module

Handles quality filtering, validation, and confidence scoring for the HotMem system.
Provides comprehensive quality assessment and filtering mechanisms.

Author: SOLID Refactoring
"""

import os
import time
from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict
import re
import statistics

from loguru import logger

class MemoryQuality:
    """
    Handles quality assessment and filtering for the HotMem system.
    
    Responsibilities:
    - Triple quality assessment and scoring
    - Confidence calculation and validation
    - Noise filtering and nonsense detection
    - Quality metrics and reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Quality thresholds
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.max_triple_length = self.config.get('max_triple_length', 100)
        self.min_entity_length = self.config.get('min_entity_length', 2)
        
        # Quality patterns
        self.nonsense_patterns = self._initialize_nonsense_patterns()
        self.quality_indicators = self._initialize_quality_indicators()
        
        # Performance tracking
        self.quality_stats = defaultdict(int)
        self.filtering_stats = defaultdict(int)
    
    def _initialize_nonsense_patterns(self) -> List[Tuple[str, str, str]]:
        """Initialize patterns to filter out nonsense triples."""
        return [
            # Meaningless conjunctions
            (r'^(is|was|are|were|be)$', r'^and$', r'^(is|was|are|were|be)$'),
            (r'^and$', r'^and$', r'.*'),
            (r'.*', r'^and$', r'^and$'),
            
            # Reflexive patterns
            (r'^(i|me|my|myself)$', r'.*', r'^(i|me|my|myself)$'),
            (r'^(you|your|yourself)$', r'.*', r'^(you|your|yourself)$'),
            
            # Generic verbs with generic objects
            (r'.*', r'^(is|are|was|were|be)$', r'.*'),
            (r'.*', r'^(have|has|had)$', r'.*'),
            (r'.*', r'^(do|does|did)$', r'.*'),
            
            # Questions as statements
            (r'^(what|where|when|why|how)$', r'.*', r'.*'),
            (r'.*', r'.*', r'^(what|where|when|why|how)$'),
            
            # Demonstrative pronouns
            (r'.*', r'.*', r'^(this|that|these|those)$'),
            (r'^(this|that|these|those)$', r'.*', r'.*'),
        ]
    
    def _initialize_quality_indicators(self) -> Dict[str, Any]:
        """Initialize quality indicators and scoring rules."""
        return {
            'high_confidence_relations': {
                'name', 'works_at', 'teach_at', 'live_in', 'has', 'owns',
                'born_in', 'studied_at', 'married_to', 'parent_of'
            },
            'low_confidence_relations': {
                'is', 'are', 'was', 'were', 'be', 'have', 'has', 'do', 'does',
                'say', 'said', 'tell', 'told', 'think', 'thought'
            },
            'entity_quality_indicators': {
                'proper_nouns': 0.2,  # Bonus for proper nouns
                'specific_entities': 0.3,  # Bonus for specific entities
                'length_appropriate': 0.1,  # Bonus for appropriate length
            },
            'context_boosters': {
                'family_context': 0.1,  # Boost for family-related facts
                'work_context': 0.1,  # Boost for work-related facts
                'location_context': 0.1,  # Boost for location-related facts
            }
        }
    
    def assess_triple_quality(self, subject: str, relation: str, object_: str,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Assess the quality of a triple.
        
        Args:
            subject: Subject entity
            relation: Relation type
            object_: Object entity
            context: Optional context information
            
        Returns:
            Dictionary with quality assessment details
        """
        assessment = {
            'triple': (subject, relation, object_),
            'confidence': 0.0,
            'issues': [],
            'quality_score': 0.0,
            'should_filter': False
        }
        
        # Basic validation
        if not self._validate_triple_basic(subject, relation, object_, assessment):
            assessment['should_filter'] = True
            return assessment
        
        # Nonsense pattern detection
        if self._matches_nonsense_pattern(subject, relation, object_):
            assessment['issues'].append('matches_nonsense_pattern')
            assessment['should_filter'] = True
            return assessment
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(subject, relation, object_, context or {})
        assessment['quality_score'] = quality_score
        
        # Apply context-based adjustments
        if context:
            quality_score = self._apply_context_adjustments(quality_score, subject, relation, object_, context)
        
        assessment['confidence'] = quality_score
        
        # Determine if should be filtered
        assessment['should_filter'] = quality_score < self.min_confidence
        
        # Update statistics
        self._update_quality_stats(assessment)
        
        return assessment
    
    def _validate_triple_basic(self, subject: str, relation: str, object_: str,
                             assessment: Dict[str, Any]) -> bool:
        """Perform basic validation of a triple."""
        if not subject or not relation or not object_:
            assessment['issues'].append('empty_component')
            return False
        
        if len(subject) > self.max_triple_length or len(relation) > self.max_triple_length or len(object_) > self.max_triple_length:
            assessment['issues'].append('too_long')
            return False
        
        if len(subject) < self.min_entity_length or len(object_) < self.min_entity_length:
            assessment['issues'].append('too_short')
            return False
        
        # Check if subject and object are the same (reflexive)
        if subject.lower() == object_.lower():
            assessment['issues'].append('reflexive')
            return False
        
        return True
    
    def _matches_nonsense_pattern(self, subject: str, relation: str, object_: str) -> bool:
        """Check if triple matches any nonsense pattern."""
        s_lower = subject.lower()
        r_lower = relation.lower()
        o_lower = object_.lower()
        
        for s_pattern, r_pattern, o_pattern in self.nonsense_patterns:
            if (re.match(s_pattern, s_lower) and 
                re.match(r_pattern, r_lower) and 
                re.match(o_pattern, o_lower)):
                return True
        
        return False
    
    def _calculate_quality_score(self, subject: str, relation: str, object_: str,
                               context: Dict[str, Any]) -> float:
        """Calculate base quality score for a triple."""
        score = 0.0
        
        # Relation-based scoring
        relation_lower = relation.lower()
        if relation_lower in self.quality_indicators['high_confidence_relations']:
            score += 0.6
        elif relation_lower in self.quality_indicators['low_confidence_relations']:
            score += 0.2
        else:
            score += 0.4  # Default for unknown relations
        
        # Entity quality scoring
        score += self._score_entity_quality(subject)
        score += self._score_entity_quality(object_)
        
        # Specificity bonus
        score += self._score_specificity(subject, relation, object_)
        
        return min(score, 1.0)
    
    def _score_entity_quality(self, entity: str) -> float:
        """Score the quality of an entity."""
        score = 0.0
        
        # Proper noun detection (capitalized words)
        if entity and entity[0].isupper():
            score += self.quality_indicators['entity_quality_indicators']['proper_nouns']
        
        # Length appropriateness
        if 2 <= len(entity) <= 50:
            score += self.quality_indicators['entity_quality_indicators']['length_appropriate']
        
        # Specific entity indicators
        entity_lower = entity.lower()
        specific_indicators = ['company', 'university', 'school', 'city', 'country', 'person']
        for indicator in specific_indicators:
            if indicator in entity_lower:
                score += self.quality_indicators['entity_quality_indicators']['specific_entities']
                break
        
        return score
    
    def _score_specificity(self, subject: str, relation: str, object_: str) -> float:
        """Score based on specificity of the triple."""
        score = 0.0
        
        # Specific relations that indicate concrete facts
        specific_relations = ['works_at', 'teach_at', 'live_in', 'born_in', 'studied_at']
        if relation.lower() in specific_relations:
            score += 0.2
        
        # Named entities (contains capital letters)
        if (any(c.isupper() for c in subject) and 
            any(c.isupper() for c in object_)):
            score += 0.1
        
        return score
    
    def _apply_context_adjustments(self, base_score: float, subject: str, relation: str,
                                 object_: str, context: Dict[str, Any]) -> float:
        """Apply context-based adjustments to quality score."""
        adjusted_score = base_score
        
        # Conversation context
        conversation_text = context.get('conversation_text', '').lower()
        
        # Family context boost
        family_keywords = ['family', 'mother', 'father', 'sister', 'brother', 'son', 'daughter']
        if any(keyword in conversation_text for keyword in family_keywords):
            adjusted_score += self.quality_indicators['context_boosters']['family_context']
        
        # Work context boost
        work_keywords = ['work', 'job', 'company', 'office', 'colleague']
        if any(keyword in conversation_text for keyword in work_keywords):
            adjusted_score += self.quality_indicators['context_boosters']['work_context']
        
        # Location context boost
        location_keywords = ['live', 'city', 'country', 'home', 'address']
        if any(keyword in conversation_text for keyword in location_keywords):
            adjusted_score += self.quality_indicators['context_boosters']['location_context']
        
        return min(adjusted_score, 1.0)
    
    def filter_triples(self, triples: List[Tuple[str, str, str]], 
                      context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str, str, float]]:
        """
        Filter a list of triples based on quality.
        
        Args:
            triples: List of (subject, relation, object) tuples
            context: Optional context information
            
        Returns:
            List of (subject, relation, object, confidence) tuples
        """
        filtered_triples = []
        
        for subject, relation, object_ in triples:
            assessment = self.assess_triple_quality(subject, relation, object_, context)
            
            if not assessment['should_filter']:
                filtered_triples.append((subject, relation, object_, assessment['confidence']))
                self.filtering_stats['passed'] += 1
            else:
                self.filtering_stats['filtered'] += 1
                logger.debug(f"Filtered triple: {subject} {relation} {object_} - {assessment['issues']}")
        
        logger.info(f"Filtered {len(triples)} triples -> {len(filtered_triples)} passed")
        return filtered_triples
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Get quality assessment statistics."""
        stats = dict(self.quality_stats)
        stats.update(dict(self.filtering_stats))
        
        total_processed = stats.get('total_processed', 0)
        if total_processed > 0:
            stats['pass_rate'] = stats.get('passed', 0) / total_processed
            stats['filter_rate'] = stats.get('filtered', 0) / total_processed
        
        return stats
    
    def _update_quality_stats(self, assessment: Dict[str, Any]) -> None:
        """Update quality statistics."""
        self.quality_stats['total_processed'] += 1
        
        if assessment['should_filter']:
            self.quality_stats['filtered'] += 1
        else:
            self.quality_stats['passed'] += 1
        
        # Track quality distribution
        score = assessment['quality_score']
        if score >= 0.8:
            self.quality_stats['high_quality'] += 1
        elif score >= 0.6:
            self.quality_stats['medium_quality'] += 1
        else:
            self.quality_stats['low_quality'] += 1
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.quality_stats.clear()
        self.filtering_stats.clear()
        logger.info("Quality statistics reset")
    
    def explain_quality_decision(self, subject: str, relation: str, object_: str) -> Dict[str, Any]:
        """
        Provide detailed explanation for quality decision.
        
        Args:
            subject: Subject entity
            relation: Relation type
            object_: Object entity
            
        Returns:
            Dictionary with detailed explanation
        """
        assessment = self.assess_triple_quality(subject, relation, object_)
        
        explanation = {
            'triple': (subject, relation, object_),
            'final_decision': 'keep' if not assessment['should_filter'] else 'filter',
            'confidence': assessment['confidence'],
            'quality_score': assessment['quality_score'],
            'issues_found': assessment['issues'],
            'scoring_breakdown': {
                'relation_score': self._score_relation(relation),
                'subject_quality': self._score_entity_quality(subject),
                'object_quality': self._score_entity_quality(object_),
                'specificity_bonus': self._score_specificity(subject, relation, object_),
            },
            'threshold': self.min_confidence
        }
        
        return explanation
    
    def _score_relation(self, relation: str) -> float:
        """Score a relation based on predefined quality indicators."""
        relation_lower = relation.lower()
        
        if relation_lower in self.quality_indicators['high_confidence_relations']:
            return 0.6
        elif relation_lower in self.quality_indicators['low_confidence_relations']:
            return 0.2
        else:
            return 0.4  # Default for unknown relations