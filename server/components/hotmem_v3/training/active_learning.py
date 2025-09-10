"""
HotMem v3 Active Learning System
Continuously improves the model by learning from usage patterns and corrections

This system implements:
1. Pattern detection in extraction errors
2. Automatic dataset generation from user interactions
3. Confidence-based uncertainty sampling
4. Continuous model improvement loop
5. User feedback integration
"""

import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import asyncio
from pathlib import Path
import hashlib
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractionError:
    """Represents an extraction error or correction"""
    original_text: str
    original_extraction: Dict[str, Any]
    corrected_extraction: Dict[str, Any]
    confidence: float
    timestamp: float
    error_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class Pattern:
    """Represents a detected pattern in errors or successes"""
    pattern_type: str  # 'entity_error', 'relation_error', 'domain_gap', etc.
    pattern_data: Dict[str, Any]
    frequency: int
    confidence: float
    first_seen: float
    last_seen: float
    examples: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LearningExample:
    """Represents a new learning example generated from user data"""
    text: str
    entities: List[str]
    relations: List[Dict[str, Any]]
    confidence: float
    source: str  # 'user_correction', 'pattern_detection', 'uncertainty_sampling'
    metadata: Dict[str, Any]
    timestamp: float

class PatternDetector:
    """Detects patterns in extraction errors and successes"""
    
    def __init__(self, min_frequency: int = 3, confidence_threshold: float = 0.7):
        self.min_frequency = min_frequency
        self.confidence_threshold = confidence_threshold
        
        # Pattern storage
        self.patterns = []
        self.error_history = deque(maxlen=1000)
        self.success_history = deque(maxlen=1000)
        
        # Pattern type detectors
        self.detectors = {
            'entity_boundary_error': self._detect_entity_boundary_errors,
            'relation_type_error': self._detect_relation_type_errors,
            'domain_gap': self._detect_domain_gaps,
            'confidence_miscalibration': self._detect_confidence_miscalibration,
            'context_dependency': self._detect_context_dependencies
        }
    
    def add_extraction_result(self, text: str, extraction: Dict[str, Any], 
                            confidence: float, is_correct: bool = True):
        """Add extraction result for pattern analysis"""
        
        result = {
            'text': text,
            'extraction': extraction,
            'confidence': confidence,
            'timestamp': time.time(),
            'is_correct': is_correct
        }
        
        if is_correct:
            self.success_history.append(result)
        else:
            self.error_history.append(result)
        
        # Run pattern detection
        self._detect_patterns()
    
    def add_user_correction(self, error: ExtractionError):
        """Add user correction for pattern learning"""
        
        self.error_history.append({
            'text': error.original_text,
            'extraction': error.original_extraction,
            'confidence': error.confidence,
            'timestamp': error.timestamp,
            'is_correct': False,
            'correction': error.corrected_extraction,
            'error_type': error.error_type
        })
        
        # Run pattern detection
        self._detect_patterns()
    
    def _detect_patterns(self):
        """Run all pattern detectors"""
        
        for pattern_type, detector in self.detectors.items():
            try:
                new_patterns = detector()
                for pattern in new_patterns:
                    self._add_pattern(pattern)
            except Exception as e:
                logger.error(f"Pattern detector {pattern_type} failed: {e}")
    
    def _detect_entity_boundary_errors(self) -> List[Pattern]:
        """Detect patterns in entity boundary errors"""
        
        patterns = []
        boundary_errors = defaultdict(list)
        
        # Analyze entity boundary errors
        for error in self.error_history:
            if 'correction' not in error:
                continue
            
            original_entities = error['extraction'].get('entities', [])
            corrected_entities = error['correction'].get('entities', [])
            
            # Find boundary differences
            for orig_ent in original_entities:
                for corr_ent in corrected_entities:
                    if self._is_entity_boundary_error(orig_ent, corr_ent, error['text']):
                        pattern_key = f"boundary_{len(orig_ent)}_{len(corr_ent)}"
                        boundary_errors[pattern_key].append(error)
        
        # Create patterns for frequent boundary errors
        for pattern_key, errors in boundary_errors.items():
            if len(errors) >= self.min_frequency:
                pattern = Pattern(
                    pattern_type='entity_boundary_error',
                    pattern_data={
                        'pattern_key': pattern_key,
                        'error_examples': [e['text'] for e in errors[:5]]
                    },
                    frequency=len(errors),
                    confidence=len(errors) / max(len(self.error_history), 1),
                    first_seen=min(e['timestamp'] for e in errors),
                    last_seen=max(e['timestamp'] for e in errors),
                    examples=errors[:3]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_relation_type_errors(self) -> List[Pattern]:
        """Detect patterns in relation type errors"""
        
        patterns = []
        relation_errors = defaultdict(list)
        
        for error in self.error_history:
            if 'correction' not in error:
                continue
            
            original_relations = error['extraction'].get('relations', [])
            corrected_relations = error['correction'].get('relations', [])
            
            # Find relation type mismatches
            for orig_rel in original_relations:
                for corr_rel in corrected_relations:
                    if (orig_rel.get('subject') == corr_rel.get('subject') and
                        orig_rel.get('object') == corr_rel.get('object') and
                        orig_rel.get('predicate') != corr_rel.get('predicate')):
                        
                        pattern_key = f"rel_type_{orig_rel.get('predicate')}_{corr_rel.get('predicate')}"
                        relation_errors[pattern_key].append(error)
        
        # Create patterns
        for pattern_key, errors in relation_errors.items():
            if len(errors) >= self.min_frequency:
                pattern = Pattern(
                    pattern_type='relation_type_error',
                    pattern_data={
                        'pattern_key': pattern_key,
                        'mismatches': [(e['extraction']['relations'][0].get('predicate'), 
                                       e['correction']['relations'][0].get('predicate')) 
                                      for e in errors[:3]]
                    },
                    frequency=len(errors),
                    confidence=len(errors) / max(len(self.error_history), 1),
                    first_seen=min(e['timestamp'] for e in errors),
                    last_seen=max(e['timestamp'] for e in errors),
                    examples=errors[:3]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_domain_gaps(self) -> List[Pattern]:
        """Detect domain gaps where the model performs poorly"""
        
        patterns = []
        domain_errors = defaultdict(list)
        
        # Analyze errors by domain (simple keyword-based domain detection)
        domain_keywords = {
            'technical': ['code', 'programming', 'software', 'algorithm', 'api'],
            'business': ['company', 'revenue', 'market', 'strategy', 'investment'],
            'personal': ['I', 'my', 'me', 'family', 'home', 'personal'],
            'academic': ['research', 'study', 'university', 'paper', 'theory']
        }
        
        for error in self.error_history:
            text = error['text'].lower()
            
            for domain, keywords in domain_keywords.items():
                if any(keyword in text for keyword in keywords):
                    domain_errors[domain].append(error)
        
        # Create patterns for domains with high error rates
        for domain, errors in domain_errors.items():
            if len(errors) >= self.min_frequency:
                pattern = Pattern(
                    pattern_type='domain_gap',
                    pattern_data={
                        'domain': domain,
                        'error_rate': len(errors) / (len(errors) + len([s for s in self.success_history if any(k in s['text'].lower() for k in domain_keywords[domain])]))
                    },
                    frequency=len(errors),
                    confidence=len(errors) / max(len(self.error_history), 1),
                    first_seen=min(e['timestamp'] for e in errors),
                    last_seen=max(e['timestamp'] for e in errors),
                    examples=errors[:3]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_confidence_miscalibration(self) -> List[Pattern]:
        """Detect confidence miscalibration patterns"""
        
        patterns = []
        
        # Group by confidence ranges
        confidence_ranges = {
            'high_confidence_errors': [],
            'low_confidence_successes': []
        }
        
        for error in self.error_history:
            if error['confidence'] > 0.8:
                confidence_ranges['high_confidence_errors'].append(error)
        
        for success in self.success_history:
            if success['confidence'] < 0.5:
                confidence_ranges['low_confidence_successes'].append(success)
        
        # Create patterns for miscalibration
        if len(confidence_ranges['high_confidence_errors']) >= self.min_frequency:
            pattern = Pattern(
                pattern_type='confidence_miscalibration',
                pattern_data={
                    'miscalibration_type': 'overconfidence',
                    'avg_confidence': np.mean([e['confidence'] for e in confidence_ranges['high_confidence_errors']])
                },
                frequency=len(confidence_ranges['high_confidence_errors']),
                confidence=len(confidence_ranges['high_confidence_errors']) / max(len(self.error_history), 1),
                first_seen=min(e['timestamp'] for e in confidence_ranges['high_confidence_errors']),
                last_seen=max(e['timestamp'] for e in confidence_ranges['high_confidence_errors']),
                examples=confidence_ranges['high_confidence_errors'][:3]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_context_dependencies(self) -> List[Pattern]:
        """Detect context-dependent extraction errors"""
        
        patterns = []
        context_errors = defaultdict(list)
        
        for error in self.error_history:
            # Simple context analysis based on text length and complexity
            text_length = len(error['text'])
            complexity = len(error['text'].split('.'))
            
            context_key = f"length_{text_length//50}_complexity_{complexity}"
            context_errors[context_key].append(error)
        
        # Create patterns for context-dependent errors
        for context_key, errors in context_errors.items():
            if len(errors) >= self.min_frequency:
                pattern = Pattern(
                    pattern_type='context_dependency',
                    pattern_data={
                        'context_key': context_key,
                        'avg_length': np.mean([len(e['text']) for e in errors])
                    },
                    frequency=len(errors),
                    confidence=len(errors) / max(len(self.error_history), 1),
                    first_seen=min(e['timestamp'] for e in errors),
                    last_seen=max(e['timestamp'] for e in errors),
                    examples=errors[:3]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _is_entity_boundary_error(self, orig_ent: str, corr_ent: str, text: str) -> bool:
        """Check if this is an entity boundary error"""
        
        # Simple heuristic: check if one entity is substring of the other
        # or if they differ by prepositions/articles
        orig_lower = orig_ent.lower()
        corr_lower = corr_ent.lower()
        
        # Remove common articles/prepositions
        articles = ['the', 'a', 'an', 'of', 'in', 'at', 'on']
        
        orig_clean = ' '.join(w for w in orig_lower.split() if w not in articles)
        corr_clean = ' '.join(w for w in corr_lower.split() if w not in articles)
        
        return (orig_clean in corr_clean or corr_clean in orig_clean or 
                orig_clean.replace(' ', '') == corr_clean.replace(' ', ''))
    
    def _add_pattern(self, pattern: Pattern):
        """Add a new pattern or update existing one"""
        
        # Check if similar pattern already exists
        for existing in self.patterns:
            if (existing.pattern_type == pattern.pattern_type and
                existing.pattern_data == pattern.pattern_data):
                
                # Update existing pattern
                existing.frequency += pattern.frequency
                existing.confidence = max(existing.confidence, pattern.confidence)
                existing.last_seen = max(existing.last_seen, pattern.last_seen)
                existing.examples.extend(pattern.examples)
                return
        
        # Add new pattern
        self.patterns.append(pattern)
        
        # Keep only recent patterns
        if len(self.patterns) > 100:
            self.patterns = sorted(self.patterns, key=lambda p: p.last_seen)[-100:]
    
    def get_significant_patterns(self, min_confidence: float = 0.5) -> List[Pattern]:
        """Get patterns that meet significance threshold"""
        
        return [p for p in self.patterns 
                if p.confidence >= min_confidence and p.frequency >= self.min_frequency]

class UncertaintySampler:
    """Samples uncertain examples for active learning"""
    
    def __init__(self, confidence_threshold: float = 0.7, max_samples: int = 100):
        self.confidence_threshold = confidence_threshold
        self.max_samples = max_samples
        self.candidate_pool = deque(maxlen=1000)
    
    def add_candidate(self, text: str, extraction: Dict[str, Any], confidence: float):
        """Add a candidate example for uncertainty sampling"""
        
        if confidence < self.confidence_threshold:
            self.candidate_pool.append({
                'text': text,
                'extraction': extraction,
                'confidence': confidence,
                'timestamp': time.time(),
                'uncertainty_score': 1.0 - confidence
            })
    
    def get_samples(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Get samples for active learning"""
        
        if not self.candidate_pool:
            return []
        
        # Sort by uncertainty score
        sorted_candidates = sorted(
            self.candidate_pool, 
            key=lambda x: x['uncertainty_score'], 
            reverse=True
        )
        
        # Return top samples
        return sorted_candidates[:min(batch_size, self.max_samples)]

class ActiveLearningSystem:
    """Main active learning system for HotMem v3"""
    
    def __init__(self, model_path: Optional[str] = None, data_dir: str = "./active_learning_data"):
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pattern_detector = PatternDetector()
        self.uncertainty_sampler = UncertaintySampler()
        
        # Learning data storage
        self.corrections = []
        self.learning_examples = []
        self.training_history = []
        
        # Active learning state
        self.last_training_time = 0
        self.min_training_interval = 3600  # 1 hour
        self.training_threshold = 50  # Min examples for training
        
        # Performance tracking
        self.improvement_metrics = defaultdict(list)
        
        # Configuration flags
        self.enable_pattern_detection = False
        self.enable_uncertainty_sampling = False
        self.confidence_threshold = 0.7
    
    def configure(self, enable_pattern_detection: bool = True, enable_uncertainty_sampling: bool = True, 
                 confidence_threshold: float = 0.7):
        """Configure the active learning system"""
        self.enable_pattern_detection = enable_pattern_detection
        self.enable_uncertainty_sampling = enable_uncertainty_sampling
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"Active learning configured: pattern_detection={enable_pattern_detection}, "
                    f"uncertainty_sampling={enable_uncertainty_sampling}, threshold={confidence_threshold}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        return {
            'total_corrections': len(self.corrections),
            'total_examples': len(self.learning_examples),
            'training_sessions': len(self.training_history),
            'last_training_time': self.last_training_time,
            'pattern_detection_enabled': self.enable_pattern_detection,
            'uncertainty_sampling_enabled': self.enable_uncertainty_sampling,
            'confidence_threshold': self.confidence_threshold,
            'improvement_metrics': dict(self.improvement_metrics)
        }
    
    def add_extraction_result(self, text: str, extraction: Dict[str, Any], 
                            confidence: float, is_correct: bool = True,
                            user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Add extraction result for learning"""
        
        # Add to pattern detector
        self.pattern_detector.add_extraction_result(text, extraction, confidence, is_correct)
        
        # Add to uncertainty sampler if uncertain
        if not is_correct or confidence < 0.7:
            self.uncertainty_sampler.add_candidate(text, extraction, confidence)
    
    def add_user_correction(self, original_text: str, original_extraction: Dict[str, Any],
                         corrected_extraction: Dict[str, Any], confidence: float,
                         error_type: str = "user_correction",
                         user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Add user correction for learning"""
        
        error = ExtractionError(
            original_text=original_text,
            original_extraction=original_extraction,
            corrected_extraction=corrected_extraction,
            confidence=confidence,
            timestamp=time.time(),
            error_type=error_type,
            user_id=user_id,
            session_id=session_id
        )
        
        self.corrections.append(error)
        self.pattern_detector.add_user_correction(error)
        
        # Generate learning example from correction
        learning_example = LearningExample(
            text=original_text,
            entities=corrected_extraction.get('entities', []),
            relations=corrected_extraction.get('relations', []),
            confidence=1.0,  # User corrections are high confidence
            source='user_correction',
            metadata={
                'error_type': error_type,
                'user_id': user_id,
                'session_id': session_id,
                'original_confidence': confidence
            },
            timestamp=time.time()
        )
        
        self.learning_examples.append(learning_example)
        
        logger.info(f"Added user correction: {len(corrected_extraction.get('entities', []))} entities, "
                   f"{len(corrected_extraction.get('relations', []))} relations")
    
    def generate_learning_examples(self) -> List[LearningExample]:
        """Generate learning examples from patterns and uncertainty sampling"""
        
        generated_examples = []
        
        # Generate examples from patterns
        patterns = self.pattern_detector.get_significant_patterns()
        
        for pattern in patterns:
            if pattern.pattern_type == 'entity_boundary_error':
                examples = self._generate_entity_boundary_examples(pattern)
                generated_examples.extend(examples)
            
            elif pattern.pattern_type == 'relation_type_error':
                examples = self._generate_relation_type_examples(pattern)
                generated_examples.extend(examples)
            
            elif pattern.pattern_type == 'domain_gap':
                examples = self._generate_domain_examples(pattern)
                generated_examples.extend(examples)
        
        # Add uncertainty samples
        uncertain_samples = self.uncertainty_sampler.get_samples(20)
        for sample in uncertain_samples:
            example = LearningExample(
                text=sample['text'],
                entities=sample['extraction'].get('entities', []),
                relations=sample['extraction'].get('relations', []),
                confidence=0.5,  # Medium confidence for uncertain samples
                source='uncertainty_sampling',
                metadata={'original_confidence': sample['confidence']},
                timestamp=time.time()
            )
            generated_examples.append(example)
        
        return generated_examples
    
    def _generate_entity_boundary_examples(self, pattern: Pattern) -> List[LearningExample]:
        """Generate examples for entity boundary patterns"""
        
        examples = []
        
        # Create synthetic examples based on pattern
        pattern_data = pattern.pattern_data.get('pattern_key', '')
        
        # This is a simplified example generation
        # In practice, you'd use more sophisticated generation methods
        synthetic_examples = [
            {
                'text': "Apple Inc. is headquartered in Cupertino California",
                'entities': ["Apple Inc.", "Cupertino, California"],
                'relations': [{'subject': 'Apple Inc.', 'predicate': 'headquartered_in', 'object': 'Cupertino, California'}]
            },
            {
                'text': "John Smith works at Microsoft Corporation",
                'entities': ["John Smith", "Microsoft Corporation"],
                'relations': [{'subject': 'John Smith', 'predicate': 'works_at', 'object': 'Microsoft Corporation'}]
            }
        ]
        
        for example_data in synthetic_examples:
            example = LearningExample(
                text=example_data['text'],
                entities=example_data['entities'],
                relations=example_data['relations'],
                confidence=0.8,
                source='pattern_detection',
                metadata={
                    'pattern_type': pattern.pattern_type,
                    'pattern_data': pattern.pattern_data
                },
                timestamp=time.time()
            )
            examples.append(example)
        
        return examples
    
    def _generate_relation_type_examples(self, pattern: Pattern) -> List[LearningExample]:
        """Generate examples for relation type patterns"""
        
        examples = []
        
        # Extract relation type mappings from pattern
        mismatches = pattern.pattern_data.get('mismatches', [])
        
        for wrong_type, correct_type in mismatches[:5]:  # Limit examples
            # Create example with correct relation type
            example = LearningExample(
                text=f"Example sentence showing {correct_type} relation",
                entities=["Entity1", "Entity2"],
                relations=[{'subject': 'Entity1', 'predicate': correct_type, 'object': 'Entity2'}],
                confidence=0.9,
                source='pattern_detection',
                metadata={
                    'pattern_type': pattern.pattern_type,
                    'wrong_type': wrong_type,
                    'correct_type': correct_type
                },
                timestamp=time.time()
            )
            examples.append(example)
        
        return examples
    
    def _generate_domain_examples(self, pattern: Pattern) -> List[LearningExample]:
        """Generate examples for domain gaps"""
        
        examples = []
        domain = pattern.pattern_data.get('domain', 'technical')
        
        # Create domain-specific examples
        domain_examples = {
            'technical': [
                {
                    'text': "The neural network uses backpropagation algorithm for optimization",
                    'entities': ["neural network", "backpropagation algorithm", "optimization"],
                    'relations': [{'subject': 'neural network', 'predicate': 'uses', 'object': 'backpropagation algorithm'}]
                }
            ],
            'business': [
                {
                    'text': "The company increased its market share in the technology sector",
                    'entities': ["company", "market share", "technology sector"],
                    'relations': [{'subject': 'company', 'predicate': 'increased', 'object': 'market share'}]
                }
            ]
        }
        
        for example_data in domain_examples.get(domain, []):
            example = LearningExample(
                text=example_data['text'],
                entities=example_data['entities'],
                relations=example_data['relations'],
                confidence=0.9,
                source='pattern_detection',
                metadata={
                    'pattern_type': pattern.pattern_type,
                    'domain': domain
                },
                timestamp=time.time()
            )
            examples.append(example)
        
        return examples
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time - self.last_training_time < self.min_training_interval:
            return False
        
        # Check if we have enough learning examples
        total_examples = len(self.learning_examples) + len(self.generate_learning_examples())
        
        return total_examples >= self.training_threshold
    
    def prepare_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data for model update"""
        
        # Combine user corrections and generated examples
        all_examples = []
        
        # Add user corrections (highest priority)
        for example in self.learning_examples:
            training_example = {
                'instruction': 'Extract entities and relations as JSON',
                'input': example.text,
                'output': json.dumps({
                    'entities': example.entities,
                    'relations': example.relations,
                    'confidence': example.confidence
                }, indent=2),
                'source': example.source,
                'metadata': example.metadata
            }
            all_examples.append(training_example)
        
        # Add generated examples
        generated_examples = self.generate_learning_examples()
        for example in generated_examples:
            training_example = {
                'instruction': 'Extract entities and relations as JSON',
                'input': example.text,
                'output': json.dumps({
                    'entities': example.entities,
                    'relations': example.relations,
                    'confidence': example.confidence
                }, indent=2),
                'source': example.source,
                'metadata': example.metadata
            }
            all_examples.append(training_example)
        
        return all_examples
    
    def save_learning_data(self):
        """Save learning data to disk"""
        
        # Save corrections
        corrections_file = self.data_dir / "corrections.json"
        with open(corrections_file, 'w') as f:
            json.dump([{
                'original_text': c.original_text,
                'original_extraction': c.original_extraction,
                'corrected_extraction': c.corrected_extraction,
                'confidence': c.confidence,
                'timestamp': c.timestamp,
                'error_type': c.error_type,
                'user_id': c.user_id,
                'session_id': c.session_id
            } for c in self.corrections], f, indent=2)
        
        # Save learning examples
        examples_file = self.data_dir / "learning_examples.json"
        with open(examples_file, 'w') as f:
            json.dump([{
                'text': e.text,
                'entities': e.entities,
                'relations': e.relations,
                'confidence': e.confidence,
                'source': e.source,
                'metadata': e.metadata,
                'timestamp': e.timestamp
            } for e in self.learning_examples], f, indent=2)
        
        # Save patterns
        patterns_file = self.data_dir / "patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump([{
                'pattern_type': p.pattern_type,
                'pattern_data': p.pattern_data,
                'frequency': p.frequency,
                'confidence': p.confidence,
                'first_seen': p.first_seen,
                'last_seen': p.last_seen
            } for p in self.pattern_detector.patterns], f, indent=2)
        
        logger.info(f"Learning data saved to {self.data_dir}")
    
    def load_learning_data(self):
        """Load learning data from disk"""
        
        # Load corrections
        corrections_file = self.data_dir / "corrections.json"
        if corrections_file.exists():
            with open(corrections_file, 'r') as f:
                corrections_data = json.load(f)
                self.corrections = [
                    ExtractionError(
                        original_text=c['original_text'],
                        original_extraction=c['original_extraction'],
                        corrected_extraction=c['corrected_extraction'],
                        confidence=c['confidence'],
                        timestamp=c['timestamp'],
                        error_type=c['error_type'],
                        user_id=c.get('user_id'),
                        session_id=c.get('session_id')
                    ) for c in corrections_data
                ]
        
        # Load learning examples
        examples_file = self.data_dir / "learning_examples.json"
        if examples_file.exists():
            with open(examples_file, 'r') as f:
                examples_data = json.load(f)
                self.learning_examples = [
                    LearningExample(
                        text=e['text'],
                        entities=e['entities'],
                        relations=e['relations'],
                        confidence=e['confidence'],
                        source=e['source'],
                        metadata=e['metadata'],
                        timestamp=e['timestamp']
                    ) for e in examples_data
                ]
        
        logger.info(f"Learning data loaded from {self.data_dir}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress"""
        
        patterns = self.pattern_detector.get_significant_patterns()
        
        return {
            'total_corrections': len(self.corrections),
            'total_learning_examples': len(self.learning_examples),
            'significant_patterns': len(patterns),
            'pattern_types': {p.pattern_type: p.frequency for p in patterns},
            'recent_corrections': len([c for c in self.corrections 
                                     if time.time() - c.timestamp < 86400]),  # Last 24 hours
            'uncertainty_pool_size': len(self.uncertainty_sampler.candidate_pool),
            'should_retrain': self.should_retrain()
        }

def main():
    """Test the active learning system"""
    
    print("🎯 HotMem v3 Active Learning System Test")
    print("=" * 50)
    
    # Initialize active learning system
    active_learning = ActiveLearningSystem()
    
    # Simulate some extraction results with errors
    test_extractions = [
        ("Apple is located in Cupertino", {"entities": ["Apple", "Cupertino"], "relations": [{"subject": "Apple", "predicate": "located_in", "object": "Cupertino"}]}, 0.9, True),
        ("Steve Jobs founded Apple", {"entities": ["Steve Jobs", "Apple"], "relations": [{"subject": "Steve Jobs", "predicate": "founded", "object": "Apple"}]}, 0.8, True),
        ("Microsoft Corporation is in Seattle", {"entities": ["Microsoft", "Seattle"], "relations": [{"subject": "Microsoft", "predicate": "in", "object": "Seattle"}]}, 0.6, False),  # Wrong entity boundary
        ("Google works on AI", {"entities": ["Google", "AI"], "relations": [{"subject": "Google", "predicate": "works_on", "object": "AI"}]}, 0.4, False),  # Low confidence
    ]
    
    print("Simulating extraction results...")
    
    for text, extraction, confidence, is_correct in test_extractions:
        print(f"Processing: '{text}' (confidence: {confidence:.2f}, correct: {is_correct})")
        active_learning.add_extraction_result(text, extraction, confidence, is_correct)
    
    # Add user corrections for errors
    print("\nAdding user corrections...")
    
    # Correction for entity boundary error
    active_learning.add_user_correction(
        original_text="Microsoft Corporation is in Seattle",
        original_extraction={"entities": ["Microsoft", "Seattle"], "relations": [{"subject": "Microsoft", "predicate": "in", "object": "Seattle"}]},
        corrected_extraction={"entities": ["Microsoft Corporation", "Seattle"], "relations": [{"subject": "Microsoft Corporation", "predicate": "located_in", "object": "Seattle"}]},
        confidence=0.6,
        error_type="entity_boundary_error"
    )
    
    # Correction for relation type error
    active_learning.add_user_correction(
        original_text="Google works on AI",
        original_extraction={"entities": ["Google", "AI"], "relations": [{"subject": "Google", "predicate": "works_on", "object": "AI"}]},
        corrected_extraction={"entities": ["Google", "AI"], "relations": [{"subject": "Google", "predicate": "develops", "object": "AI"}]},
        confidence=0.4,
        error_type="relation_type_error"
    )
    
    # Get learning summary
    print(f"\n{'='*50}")
    print("Learning Summary:")
    summary = active_learning.get_learning_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Check if retraining is needed
    if summary['should_retrain']:
        print(f"\n{'='*50}")
        print("Preparing training data for retraining...")
        training_data = active_learning.prepare_training_data()
        print(f"Generated {len(training_data)} training examples")
        
        # Show some examples
        print("\nSample training examples:")
        for i, example in enumerate(training_data[:3]):
            print(f"\nExample {i+1}:")
            print(f"  Input: {example['input']}")
            print(f"  Source: {example['source']}")
            print(f"  Metadata: {example['metadata']}")
    
    # Save learning data
    print(f"\n{'='*50}")
    print("Saving learning data...")
    active_learning.save_learning_data()
    print("✅ Learning data saved")

if __name__ == "__main__":
    main()