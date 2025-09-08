# HotMem Evolution Phase 3: Enhanced Retrieval & Context Intelligence

**Vision**: Transform LocalCat into an intelligent context-aware system that provides agents with the best possible information at every turn, leveraging all available local data sources for complex queries.

## 🎯 Executive Summary

**Current State**: Phase 2 delivered real-time corrections (67% success) and session isolation, but **only 20% of our retrieval potential is being harnessed**.

**Goal**: Unlock the remaining 80% of retrieval capability through multi-layer context assembly, intelligent information synthesis, and adaptive quality optimization.

**Expected Impact**: 
- 4x improvement in retrieval coverage (20% → 80%)
- 90%+ satisfaction rate for complex queries
- 50% better token efficiency through smart filtering
- Cross-session knowledge continuity

---

## 📊 Current Retrieval Analysis

### What's Working (20% of potential)
- ✅ **Memory bullets**: 5 structured facts per turn
- ✅ **Temporal decay**: Recent facts properly weighted (α=0.15, β=0.60)
- ✅ **Real-time corrections**: 67% success rate with UD patterns
- ✅ **Session isolation**: Clean test scenarios

### What's Missing (80% of potential)
- ❌ **Session summaries**: Rich context stored but never retrieved
- ❌ **LEANN vectors**: Semantic search capability completely unused
- ❌ **Verbatim snippets**: Exact conversation fragments inaccessible
- ❌ **Cross-session knowledge**: No inter-session fact correlation
- ❌ **Complex query handling**: Multi-clause questions poorly served
- ❌ **Information synthesis**: Raw data dumps instead of intelligent context

---

## 🏗️ Phase 3 Architecture Overview

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│ Query Analyzer  │───▶│ Multi-Layer Retrieval │───▶│ Context Intelligence │
│                 │    │                      │    │                     │
│ • Complexity    │    │ Layer 1: Facts       │    │ • Relevance scoring │
│ • Intent type   │    │ Layer 2: Summaries   │    │ • Confidence marks  │
│ • Entities      │    │ Layer 3: Semantic    │    │ • Query filtering   │
│ • Temporal      │    │ Layer 4: Verbatim    │    │ • Token optimization│
└─────────────────┘    └──────────────────────┘    └─────────────────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ Agent Context       │
                      │ Package            │
                      │                    │
                      │ • Structured facts │
                      │ • Session context  │
                      │ • Semantic links   │
                      │ • Verbatim quotes  │
                      │ • Confidence: HIGH │
                      └─────────────────────┘
```

---

## 🎯 Priority 1: Multi-Layer Retrieval Architecture (3-5 days)

### Problem Statement
Currently, only structured memory bullets reach agents. This leaves vast amounts of useful information unused:
- Session summaries with rich contextual insights
- LEANN semantic vectors for conceptual matching
- Verbatim conversation snippets for exact references
- Cross-session entity relationships

### Solution: Intelligent Multi-Layer Context Assembly

#### Layer 1: Structured Facts (Current - 20% potential)
```python
# memory_hotpath.py - Enhanced structured retrieval
class StructuredRetrieval:
    def retrieve_facts(self, query_entities, turn_id, session_id):
        """Enhanced fact retrieval with entity tracking"""
        candidates = self._gather_candidates(query_entities)
        
        # Apply temporal decay (existing)
        scored_facts = self._apply_temporal_scoring(candidates, turn_id)
        
        # NEW: Cross-session entity tracking
        related_facts = self._find_related_entities(query_entities, session_id)
        
        # Merge and rank
        all_facts = self._merge_and_rank(scored_facts, related_facts)
        
        return self._format_as_bullets(all_facts[:5])
```

#### Layer 2: Session Context (NEW - 30% potential)
```python
# session_retrieval.py - NEW component
class SessionContextRetrieval:
    def retrieve_summaries(self, query, session_id, entities):
        """Retrieve relevant session summaries via FTS"""
        # Search current session summaries
        current_summaries = self.store.search_fts(
            query, 
            filters={'session_id': session_id},
            limit=3
        )
        
        # Search related entity summaries from other sessions
        entity_summaries = []
        for entity in entities:
            cross_session = self.store.search_fts(
                entity,
                filters={'type': 'summary'},
                limit=2
            )
            entity_summaries.extend(cross_session)
        
        return self._deduplicate_and_score(current_summaries, entity_summaries)
    
    def retrieve_long_term_context(self, entities, days_back=7):
        """Get entity history across sessions"""
        entity_timeline = {}
        for entity in entities:
            timeline = self.store.get_entity_timeline(
                entity, 
                since=datetime.now() - timedelta(days=days_back)
            )
            entity_timeline[entity] = timeline
        
        return self._synthesize_entity_history(entity_timeline)
```

#### Layer 3: Semantic Understanding (NEW - 35% potential)
```python
# semantic_retrieval.py - Enhanced LEANN integration
class SemanticRetrieval:
    def __init__(self):
        self.leann_adapter = LEANNAdapter()
    
    def retrieve_semantic_matches(self, query, complexity=16, limit=5):
        """Semantic similarity search via LEANN"""
        if not self.leann_adapter.index_exists():
            logger.warning("LEANN index not available, skipping semantic search")
            return []
        
        # Query embedding
        query_vector = self.leann_adapter.embed_query(query)
        
        # Semantic search
        similar_items = self.leann_adapter.search(
            query_vector,
            complexity=complexity,
            limit=limit * 2  # Get more, filter later
        )
        
        # Filter by relevance threshold
        relevant_items = [
            item for item in similar_items 
            if item.score > 0.7  # High relevance only
        ]
        
        return relevant_items[:limit]
    
    def find_conceptual_connections(self, entities, query):
        """Find related concepts beyond exact matches"""
        conceptual_matches = []
        
        for entity in entities:
            # Find similar entities/concepts
            entity_vector = self.leann_adapter.embed_query(entity)
            similar_concepts = self.leann_adapter.search(
                entity_vector, 
                complexity=16,
                limit=3
            )
            
            # Filter for different entities (not exact matches)
            related_concepts = [
                c for c in similar_concepts 
                if c.text.lower() != entity.lower() and c.score > 0.6
            ]
            
            conceptual_matches.extend(related_concepts)
        
        return self._rank_conceptual_relevance(conceptual_matches, query)
```

#### Layer 4: Verbatim Evidence (NEW - 15% potential)
```python
# verbatim_retrieval.py - NEW component
class VerbatimRetrieval:
    def retrieve_exact_quotes(self, query, entities, session_id):
        """Find exact conversation snippets"""
        verbatim_matches = []
        
        # Search for exact phrases in conversation history
        exact_phrases = self._extract_key_phrases(query)
        
        for phrase in exact_phrases:
            matches = self.store.search_verbatim(
                phrase, 
                session_id=session_id,
                context_window=50  # 50 chars before/after
            )
            verbatim_matches.extend(matches)
        
        # Search for entity mentions with context
        for entity in entities:
            entity_mentions = self.store.search_entity_mentions(
                entity,
                session_id=session_id,
                context_window=100
            )
            verbatim_matches.extend(entity_mentions)
        
        return self._rank_by_recency_and_relevance(verbatim_matches)
    
    def get_conversation_flow_context(self, query, turn_id, window=5):
        """Get surrounding conversation context"""
        # Get conversation turns around current turn
        context_turns = self.store.get_turn_window(
            turn_id, 
            before=window, 
            after=0
        )
        
        # Filter for relevance to current query
        relevant_context = []
        for turn in context_turns:
            if self._is_contextually_relevant(turn.text, query):
                relevant_context.append(turn)
        
        return relevant_context
```

### Multi-Layer Orchestration
```python
# retrieval_orchestrator.py - NEW component  
class RetrievalOrchestrator:
    def __init__(self):
        self.structured = StructuredRetrieval()
        self.session = SessionContextRetrieval()
        self.semantic = SemanticRetrieval() 
        self.verbatim = VerbatimRetrieval()
        self.query_analyzer = QueryAnalyzer()
    
    async def retrieve_context(self, query, entities, turn_id, session_id):
        """Orchestrate multi-layer retrieval"""
        # Analyze query complexity and needs
        analysis = self.query_analyzer.analyze(query, entities)
        
        retrieval_tasks = []
        
        # Layer 1: Always get structured facts
        retrieval_tasks.append(
            self.structured.retrieve_facts(entities, turn_id, session_id)
        )
        
        # Layer 2: Session context for complex queries
        if analysis.complexity >= QueryComplexity.MEDIUM:
            retrieval_tasks.append(
                self.session.retrieve_summaries(query, session_id, entities)
            )
        
        # Layer 3: Semantic search for high complexity
        if analysis.complexity >= QueryComplexity.HIGH:
            retrieval_tasks.append(
                self.semantic.retrieve_semantic_matches(query)
            )
        
        # Layer 4: Verbatim for reference queries
        if analysis.intent_type == QueryIntent.REFERENCE:
            retrieval_tasks.append(
                self.verbatim.retrieve_exact_quotes(query, entities, session_id)
            )
        
        # Execute all retrieval layers concurrently
        layer_results = await asyncio.gather(*retrieval_tasks)
        
        # Combine and optimize results
        return self._synthesize_context(layer_results, analysis)
```

---

## 🧠 Priority 2: Context-Aware Agent Intelligence (2-3 days)

### Problem Statement
Agents receive raw information dumps without understanding:
- Which information is most relevant to their specific query
- Confidence levels and reliability of different facts
- Relationships between pieces of information
- Why specific information was included

### Solution: Intelligent Context Preparation & Delivery

#### Query-Specific Filtering & Relevance Scoring
```python
# context_intelligence.py - NEW component
class ContextIntelligence:
    def __init__(self):
        self.relevance_scorer = RelevanceScorer()
        self.confidence_assessor = ConfidenceAssessor()
        self.information_synthesizer = InformationSynthesizer()
    
    def prepare_agent_context(self, raw_retrieval_results, query_analysis):
        """Transform raw retrieval into intelligent agent context"""
        
        # Score relevance for this specific query
        scored_items = []
        for layer_name, items in raw_retrieval_results.items():
            for item in items:
                relevance_score = self.relevance_scorer.score(
                    item, 
                    query_analysis.query,
                    query_analysis.entities,
                    query_analysis.intent_type
                )
                
                confidence_score = self.confidence_assessor.assess(
                    item,
                    age=item.age,
                    source=layer_name,
                    verification_count=item.verification_count
                )
                
                scored_items.append({
                    'item': item,
                    'relevance': relevance_score,
                    'confidence': confidence_score,
                    'layer': layer_name,
                    'reasoning': f"Relevant to {query_analysis.intent_type} query about {', '.join(query_analysis.entities)}"
                })
        
        # Filter and rank by combined score
        filtered_items = [
            item for item in scored_items 
            if item['relevance'] > 0.6 and item['confidence'] > 0.5
        ]
        
        ranked_items = sorted(
            filtered_items,
            key=lambda x: (x['relevance'] * 0.7 + x['confidence'] * 0.3),
            reverse=True
        )
        
        # Synthesize into coherent context
        return self.information_synthesizer.synthesize(
            ranked_items,
            query_analysis,
            max_tokens=1000
        )
```

#### Information Synthesis & Organization
```python
class InformationSynthesizer:
    def synthesize(self, ranked_items, query_analysis, max_tokens):
        """Organize information into coherent agent context"""
        
        context_sections = {
            'immediate_facts': [],
            'background_context': [],
            'related_information': [],
            'verbatim_references': []
        }
        
        token_budget = TokenBudget(max_tokens)
        
        # Prioritize immediate facts
        for item in ranked_items:
            if item['layer'] == 'structured' and item['relevance'] > 0.8:
                if token_budget.can_add(item['item']):
                    context_sections['immediate_facts'].append(item)
                    token_budget.add(item['item'])
        
        # Add background from summaries
        for item in ranked_items:
            if item['layer'] == 'session' and item['confidence'] > 0.7:
                if token_budget.can_add(item['item']):
                    context_sections['background_context'].append(item)
                    token_budget.add(item['item'])
        
        # Add semantic connections
        for item in ranked_items:
            if item['layer'] == 'semantic' and item['relevance'] > 0.7:
                if token_budget.can_add(item['item']):
                    context_sections['related_information'].append(item)
                    token_budget.add(item['item'])
        
        # Add verbatim quotes for reference
        for item in ranked_items:
            if item['layer'] == 'verbatim' and item['confidence'] > 0.8:
                if token_budget.can_add(item['item']):
                    context_sections['verbatim_references'].append(item)
                    token_budget.add(item['item'])
        
        return self._format_agent_context(context_sections, query_analysis)
    
    def _format_agent_context(self, sections, analysis):
        """Format context for optimal agent comprehension"""
        context_parts = []
        
        # Add query context
        context_parts.append(f"## Query Analysis")
        context_parts.append(f"User is asking about: {', '.join(analysis.entities)}")
        context_parts.append(f"Query type: {analysis.intent_type}")
        context_parts.append(f"Complexity: {analysis.complexity}")
        context_parts.append("")
        
        # Immediate facts (highest priority)
        if sections['immediate_facts']:
            context_parts.append("## Immediate Facts (High Confidence)")
            for item in sections['immediate_facts']:
                confidence_marker = "🟢" if item['confidence'] > 0.8 else "🟡"
                context_parts.append(f"{confidence_marker} {item['item'].text}")
                if item.get('reasoning'):
                    context_parts.append(f"   └─ Why included: {item['reasoning']}")
            context_parts.append("")
        
        # Background context
        if sections['background_context']:
            context_parts.append("## Background Context")
            for item in sections['background_context']:
                age_marker = "📅" if item['item'].age_days < 1 else "📆"
                context_parts.append(f"{age_marker} {item['item'].text}")
            context_parts.append("")
        
        # Related information
        if sections['related_information']:
            context_parts.append("## Related Information")
            for item in sections['related_information']:
                context_parts.append(f"🔗 {item['item'].text}")
            context_parts.append("")
        
        # Verbatim references
        if sections['verbatim_references']:
            context_parts.append("## Exact References")
            for item in sections['verbatim_references']:
                context_parts.append(f'💬 "{item['item'].text}"')
            context_parts.append("")
        
        return "\n".join(context_parts)
```

#### Confidence Assessment & Uncertainty Communication
```python
class ConfidenceAssessor:
    def assess(self, item, age, source, verification_count=1):
        """Assess information confidence based on multiple factors"""
        base_confidence = 0.7
        
        # Age factor (fresher is more reliable)
        age_factor = max(0.5, 1.0 - (age / 86400))  # Decay over 24 hours
        
        # Source reliability
        source_weights = {
            'structured': 0.9,    # High reliability from fact extraction
            'session': 0.8,      # Good reliability from summaries  
            'semantic': 0.7,     # Moderate reliability from semantic matching
            'verbatim': 0.95     # Very high reliability for exact quotes
        }
        source_factor = source_weights.get(source, 0.6)
        
        # Verification factor (how many times confirmed)
        verification_factor = min(1.0, 0.6 + (verification_count * 0.1))
        
        # Combined confidence
        confidence = base_confidence * age_factor * source_factor * verification_factor
        
        return min(1.0, confidence)
```

---

## 🔄 Priority 3: Retrieval Quality Feedback Loop (1-2 days)

### Problem Statement
No mechanism to learn what information is actually useful vs noise. The system cannot improve its retrieval quality over time.

### Solution: Adaptive Retrieval Optimization

#### Usage Analytics & Quality Tracking
```python
# retrieval_analytics.py - NEW component
class RetrievalAnalytics:
    def __init__(self):
        self.usage_tracker = UsageTracker()
        self.quality_assessor = QualityAssessor()
        self.feedback_processor = FeedbackProcessor()
    
    def track_retrieval_usage(self, query, retrieved_items, agent_response):
        """Track which retrieved information was actually used"""
        
        usage_analysis = {
            'query': query,
            'retrieved_items': retrieved_items,
            'agent_response': agent_response,
            'timestamp': datetime.utcnow()
        }
        
        # Analyze which items were referenced in response
        referenced_items = []
        for item in retrieved_items:
            if self._was_item_referenced(item, agent_response):
                referenced_items.append({
                    'item_id': item.id,
                    'layer': item.layer,
                    'relevance_score': item.relevance,
                    'confidence_score': item.confidence,
                    'usage_type': self._classify_usage(item, agent_response)
                })
        
        usage_analysis['referenced_items'] = referenced_items
        
        # Calculate precision metrics
        precision = len(referenced_items) / len(retrieved_items) if retrieved_items else 0
        usage_analysis['precision'] = precision
        
        # Store for analysis
        self.usage_tracker.record_usage(usage_analysis)
        
        return usage_analysis
    
    def _was_item_referenced(self, item, agent_response):
        """Check if agent actually used the retrieved item"""
        # Simple keyword overlap for now
        item_keywords = set(item.text.lower().split())
        response_keywords = set(agent_response.lower().split())
        
        overlap = item_keywords.intersection(response_keywords)
        overlap_ratio = len(overlap) / len(item_keywords) if item_keywords else 0
        
        return overlap_ratio > 0.3  # 30% keyword overlap threshold
    
    def _classify_usage(self, item, agent_response):
        """Classify how the item was used in the response"""
        if any(quote in agent_response for quote in ['"', "'"]):
            return "direct_quote"
        elif item.text.lower() in agent_response.lower():
            return "direct_reference"
        else:
            return "conceptual_influence"
```

#### Dynamic Layer Weight Tuning
```python
class AdaptiveRetrieval:
    def __init__(self):
        self.layer_weights = {
            'structured': 1.0,
            'session': 0.8,
            'semantic': 0.6, 
            'verbatim': 0.4
        }
        self.performance_history = []
    
    def update_layer_weights(self, usage_analytics):
        """Adjust layer weights based on usage patterns"""
        
        # Analyze layer performance
        layer_performance = {}
        for layer in self.layer_weights:
            layer_items = [
                item for item in usage_analytics['referenced_items']
                if item['layer'] == layer
            ]
            
            if layer_items:
                avg_usage_quality = sum(
                    item['relevance_score'] * (1 if item['usage_type'] != 'unused' else 0)
                    for item in layer_items
                ) / len(layer_items)
                layer_performance[layer] = avg_usage_quality
            else:
                layer_performance[layer] = 0.0
        
        # Adjust weights based on performance
        total_performance = sum(layer_performance.values())
        if total_performance > 0:
            for layer in self.layer_weights:
                performance_ratio = layer_performance[layer] / total_performance
                
                # Gradual adjustment (learning rate = 0.1)
                adjustment = (performance_ratio - 0.25) * 0.1  # 0.25 = 1/4 layers baseline
                self.layer_weights[layer] = max(0.1, min(1.5, self.layer_weights[layer] + adjustment))
        
        # Log weight changes
        logger.info(f"Updated layer weights: {self.layer_weights}")
        
        return self.layer_weights
```

#### User Feedback Integration
```python
class FeedbackProcessor:
    def process_correction(self, original_query, correction_text, retrieved_items):
        """Learn from user corrections about retrieval quality"""
        
        correction_analysis = {
            'original_query': original_query,
            'correction': correction_text,
            'timestamp': datetime.utcnow()
        }
        
        # Analyze what was missing or wrong
        missing_entities = self._extract_missing_entities(correction_text)
        wrong_facts = self._identify_contradicted_facts(correction_text, retrieved_items)
        
        correction_analysis['missing_entities'] = missing_entities
        correction_analysis['contradicted_facts'] = wrong_facts
        
        # Update retrieval patterns
        self._update_entity_patterns(missing_entities)
        self._downweight_incorrect_sources(wrong_facts)
        
        return correction_analysis
    
    def _extract_missing_entities(self, correction_text):
        """Extract entities that should have been retrieved"""
        # Use same UD parsing as correction system
        doc = nlp(correction_text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'should_have_been_retrieved': True
            })
        
        return entities
    
    def _update_entity_patterns(self, missing_entities):
        """Improve entity recognition patterns"""
        for entity in missing_entities:
            # Add to entity recognition training data
            self._add_entity_pattern(entity['text'], entity['label'])
    
    def _downweight_incorrect_sources(self, wrong_facts):
        """Reduce confidence in sources that provided wrong information"""
        for fact in wrong_facts:
            # Reduce source confidence
            self._adjust_source_weight(fact['source'], -0.1)
```

---

## 🚀 Implementation Timeline

### Week 1: Multi-Layer Retrieval Foundation (Days 1-3)
**Day 1**: 
- Implement Query Analyzer and complexity classification
- Create basic Multi-Layer Retrieval Orchestra
- Set up Layer 2 (Session Context) infrastructure

**Day 2**:
- Implement Layer 3 (Semantic/LEANN) integration
- Create Layer 4 (Verbatim) retrieval system
- Build basic context synthesis

**Day 3**:
- Integration testing of all layers
- Performance optimization and caching
- Basic relevance scoring implementation

### Week 2: Context Intelligence & Feedback (Days 4-5)
**Day 4**:
- Implement Context Intelligence with relevance scoring
- Build Information Synthesizer with token budgeting
- Create Confidence Assessment system

**Day 5**:
- Implement Usage Analytics and feedback tracking
- Build Adaptive Retrieval weight tuning
- Create comprehensive test suite

### Success Metrics & Validation

#### Performance Targets
- [ ] **Simple queries**: <100ms response time (Layer 1 only)
- [ ] **Complex queries**: <300ms response time (Multi-layer)
- [ ] **Retrieval coverage**: 80%+ of available information accessible
- [ ] **Context efficiency**: 50% better token utilization

#### Quality Targets  
- [ ] **Query satisfaction**: 90%+ complex questions get relevant context
- [ ] **Precision improvement**: 40%+ reduction in irrelevant information
- [ ] **Cross-session continuity**: Entity relationships maintained across sessions
- [ ] **Adaptive learning**: Layer weights improve over 2 weeks of usage

#### User Experience Targets
- [ ] **Temporal queries**: "What happened last week?" gets comprehensive summaries
- [ ] **Relationship queries**: "How are X and Y connected?" leverages cross-session data  
- [ ] **Complex references**: "That thing we discussed about..." finds exact context
- [ ] **Learning continuity**: Agents build on previous conversations naturally

---

## 🛠️ Technical Implementation Notes

### New Files to Create
```
server/
├── retrieval/
│   ├── __init__.py
│   ├── orchestrator.py          # Multi-layer coordination
│   ├── query_analyzer.py        # Query complexity & intent analysis
│   ├── session_retrieval.py     # Layer 2: Session summaries
│   ├── semantic_retrieval.py    # Layer 3: LEANN integration  
│   ├── verbatim_retrieval.py    # Layer 4: Exact quotes
│   └── context_intelligence.py  # Agent context preparation
├── analytics/
│   ├── __init__.py
│   ├── usage_tracker.py         # Track retrieval usage
│   ├── quality_assessor.py      # Measure retrieval quality
│   ├── feedback_processor.py    # Learn from corrections
│   └── adaptive_retrieval.py    # Dynamic weight tuning
└── tests/
    ├── test_multi_layer_retrieval.py
    ├── test_context_intelligence.py
    └── test_adaptive_learning.py
```

### Integration Points
- **HotPath Processor**: Replace simple memory bullet injection with multi-layer retrieval
- **LEANN Adapter**: Enhance to support semantic query analysis
- **Memory Store**: Add verbatim storage and cross-session entity tracking
- **Correction System**: Integrate with feedback processor for learning

### Environment Configuration
```bash
# Phase 3 configuration additions to .env
HOTMEM_MULTILAYER_RETRIEVAL=true
HOTMEM_MAX_CONTEXT_TOKENS=1000
HOTMEM_SEMANTIC_THRESHOLD=0.7
HOTMEM_VERBATIM_CONTEXT_WINDOW=100
HOTMEM_ADAPTIVE_LEARNING=true
HOTMEM_USAGE_ANALYTICS=true
```

---

## 🎯 Phase 3 Success Vision

**End State**: LocalCat becomes an intelligent memory system that:

1. **Understands Complex Queries**: Multi-clause questions get comprehensive, relevant context
2. **Provides Contextual Intelligence**: Agents receive organized, confidence-marked information  
3. **Learns from Usage**: System improves retrieval quality based on actual utility
4. **Maintains Continuity**: Cross-session knowledge provides seamless conversation evolution
5. **Optimizes Resources**: Smart token usage maximizes information density

**User Experience**: 
- "What did we discuss about Potola's training last week?" → Gets structured facts + session summaries + semantic connections + exact quotes
- "How is my work situation related to my stress levels?" → Cross-session correlation with confidence scoring
- "That thing you mentioned about..." → Exact verbatim retrieval with conversation context

**Technical Achievement**:
- 80% of available information accessible through intelligent retrieval
- <300ms comprehensive context assembly for complex queries
- Self-improving system that adapts to usage patterns
- Foundation for future AI agent intelligence enhancements