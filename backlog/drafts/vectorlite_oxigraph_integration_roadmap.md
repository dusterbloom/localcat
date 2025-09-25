# VectorLiteDB and Oxigraph Integration for RDF-Star Compliant Knowledge Graphs
*Enhancing LocalCat's Memory System with Vector Retrieval and Semantic Reasoning*

## Executive Summary

This document outlines the integration of VectorLiteDB and Oxigraph into LocalCat's memory system to achieve full RDF-star compliance while maintaining the <300ms hot-path latency target. VectorLiteDB will enhance retrieval quality through vector similarity, while Oxigraph enables semantic reasoning and standardized knowledge export. Both operate in the cold path, preserving the speed-critical voice pipeline while adding powerful graph capabilities for background processing and offline analysis.

## 🎯 Strategic Value

### Why VectorLiteDB + Oxigraph Matters

**VectorLiteDB Benefits:**
- **Enhanced Retrieval**: Vector similarity for conceptual queries beyond adjacency
- **Local-First**: SQLite-based, no external dependencies
- **Performance**: Optimized for read-heavy workloads with vector indexing

**Oxigraph Benefits:**
- **RDF-Star Compliance**: Finally realizes the original vision for semantic knowledge graphs
- **SPARQL Reasoning**: Complex queries impossible with current LMDB adjacency
- **Standards-Based**: OWL/SHACL validation and interoperable data export
- **Temporal Reasoning**: RDF-star annotations for time-aware queries

### Unique Advantages for LocalCat

- **Hybrid Retrieval**: Adjacency (hot) + Vector similarity (cold) + SPARQL reasoning (offline)
- **Knowledge Portability**: Export conversations as RDF-star for external analysis
- **Semantic Validation**: SHACL constraints ensure knowledge graph consistency
- **Multi-Hop Reasoning**: "How does Alice's coffee preference relate to her work schedule?"

## 🏗️ Technical Architecture

### Current Memory Pipeline
```
Voice Input → UD Extraction → SQLite/LMDB → Retrieval → LLM Injection
                    (Hot Path: <200ms)         (Cold Path: Async)
```

### Enhanced Pipeline with VectorLiteDB + Oxigraph
```
Voice Input → UD Extraction → SQLite/LMDB → Retrieval → LLM Injection
   ↓              ↓                    ↓
   └─→ VectorLiteDB (Similarity) ─────┘
         ↓
   └─→ Oxigraph (RDF-Star Reasoning)
         ↓
   └─→ Background Consolidation & Export
```

## 📊 VectorLiteDB Integration

### Core Capabilities

#### 1. **Vector Storage & Indexing**
- **Embeddings**: Store entity/relation embeddings for similarity search
- **Hybrid Indexing**: B-tree + vector indexes for multi-modal retrieval
- **Incremental Updates**: Micro-batch vector insertions without blocking

#### 2. **Enhanced Retrieval Modes**
- **Semantic Similarity**: Find related concepts beyond exact matches
- **Conceptual Queries**: "Things related to coffee" → finds "espresso", "caffeine", "morning routine"
- **Fuzzy Matching**: Handle typos and variations in entity names

#### 3. **Cold Path Integration**
- **Async Processing**: Vector computation happens post-turn
- **Fallback Enhancement**: If adjacency returns <3 bullets, supplement with vector similarity
- **Learning Loop**: GEPA can optimize vector similarity thresholds

### Oxigraph Integration

#### 1. **RDF-Star Knowledge Graph**
- **Triple Storage**: Convert extracted facts to RDF-star triples
- **Temporal Annotations**: `<< :Alice :drinks :coffee >> :at "2025-01-15T09:30:00"^^xsd:dateTime`
- **Confidence Metadata**: `<< :fact >> :confidence 0.85 ; :source :ud_extraction`

#### 2. **Semantic Reasoning**
- **OWL Inference**: Automatic classification and property inheritance
- **SHACL Validation**: Ensure knowledge graph consistency
- **SPARQL Queries**: Complex multi-hop reasoning

#### 3. **Background Processing**
- **Consolidation**: Merge duplicate facts across sessions
- **Cross-References**: Link related concepts automatically
- **Export Pipeline**: Generate RDF-star dumps for external tools

## 🛠️ Implementation Plan

### Phase 1: VectorLiteDB Foundation (Week 1-2)

#### Database Setup
```python
# server/vector_store.py
import vectorlite
from vectorlite import VectorLite

class VectorMemoryStore:
    def __init__(self, db_path="vector_memory.db"):
        self.db = VectorLite(db_path, vectors=True)
        
        # Create tables for different embedding types
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS entity_vectors (
                entity_id TEXT PRIMARY KEY,
                vector BLOB,
                metadata TEXT,
                updated_at INTEGER
            )
        """)
        
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS relation_vectors (
                subject TEXT,
                relation TEXT, 
                object TEXT,
                vector BLOB,
                confidence REAL,
                timestamp INTEGER,
                PRIMARY KEY (subject, relation, object)
            )
        """)

    def store_entity_vector(self, entity_id: str, vector: np.ndarray, metadata: dict):
        """Store entity embedding with metadata"""
        vector_bytes = vector.tobytes()
        metadata_json = json.dumps(metadata)
        
        self.db.execute("""
            INSERT OR REPLACE INTO entity_vectors 
            (entity_id, vector, metadata, updated_at)
            VALUES (?, ?, ?, ?)
        """, (entity_id, vector_bytes, metadata_json, int(time.time())))

    def find_similar_entities(self, query_vector: np.ndarray, limit: int = 5) -> List[dict]:
        """Find entities similar to query vector"""
        return self.db.execute("""
            SELECT entity_id, metadata, 
                   vector_distance(vector, ?) as similarity
            FROM entity_vectors 
            ORDER BY similarity ASC 
            LIMIT ?
        """, (query_vector.tobytes(), limit))
```

#### Vector Computation Pipeline
```python
# server/vector_processor.py
import sentence_transformers
from sentence_transformers import SentenceTransformer

class VectorProcessor:
    def __init__(self):
        # Use lightweight local model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = VectorMemoryStore()
    
    async def process_turn_async(self, entities: List[str], triples: List[Tuple[str,str,str]]):
        """Async vector computation post-turn"""
        
        # Entity embeddings
        entity_texts = [self._entity_to_text(e) for e in entities]
        entity_vectors = self.model.encode(entity_texts, normalize_embeddings=True)
        
        for entity, vector in zip(entities, entity_vectors):
            metadata = {"type": "entity", "source": "conversation"}
            await self.vector_store.store_entity_vector(entity, vector, metadata)
        
        # Relation embeddings (combine subject + relation + object)
        for s, r, d in triples:
            relation_text = f"{s} {r} {d}"
            vector = self.model.encode([relation_text], normalize_embeddings=True)[0]
            
            await self.vector_store.store_relation_vector(s, r, d, vector, 
                                                        confidence=0.8, 
                                                        timestamp=int(time.time()))
```

### Phase 2: Oxigraph RDF-Star Graph (Week 3-4)

#### RDF-Star Triple Generation
```python
# server/rdf_graph.py
from oxigraph import Store
import rdflib
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD

class RDFStarGraph:
    def __init__(self):
        self.store = Store()
        self.graph = Graph(store=self.store)
        
        # Define namespaces
        self.LOCAT = rdflib.Namespace("https://localcat.ai/")
        self.graph.bind("locat", self.LOCAT)
    
    def add_fact_with_metadata(self, subject: str, relation: str, obj: str, 
                             confidence: float, timestamp: int, source: str):
        """Add RDF-star triple with metadata"""
        
        # Create the main triple
        s_uri = self._to_uri(subject)
        p_uri = self._to_uri(relation) 
        o_uri = self._to_uri(obj)
        
        # Create RDF-star triple with annotations
        fact = BNode()
        self.graph.add((fact, RDF.type, self.LOCAT.Fact))
        self.graph.add((fact, self.LOCAT.subject, s_uri))
        self.graph.add((fact, self.LOCAT.relation, p_uri))
        self.graph.add((fact, self.LOCAT.object, o_uri))
        self.graph.add((fact, self.LOCAT.confidence, Literal(confidence, datatype=XSD.float)))
        self.graph.add((fact, self.LOCAT.timestamp, Literal(timestamp, datatype=XSD.integer)))
        self.graph.add((fact, self.LOCAT.source, Literal(source)))
        
        # Add the annotated triple
        annotated_triple = BNode()
        self.graph.add((annotated_triple, RDF.subject, s_uri))
        self.graph.add((annotated_triple, RDF.predicate, p_uri))
        self.graph.add((annotated_triple, RDF.object, o_uri))
        self.graph.add((fact, self.LOCAT.annotates, annotated_triple))
    
    def query_similar_facts(self, entity: str, limit: int = 5) -> List[dict]:
        """SPARQL query for facts involving similar entities"""
        query = f"""
        SELECT ?fact ?subject ?relation ?object ?confidence
        WHERE {{
            ?fact locat:annotates ?triple .
            ?fact locat:confidence ?confidence .
            ?triple rdf:subject ?subject .
            ?triple rdf:predicate ?relation .
            ?triple rdf:object ?object .
            
            FILTER(?subject = <{self._to_uri(entity)}> || 
                   ?object = <{self._to_uri(entity)}>)
        }}
        ORDER BY DESC(?confidence)
        LIMIT {limit}
        """
        
        results = self.graph.query(query)
        return [dict(row) for row in results]
```

#### Background Consolidation Service
```python
# server/graph_consolidator.py
class GraphConsolidator:
    def __init__(self, rdf_graph: RDFStarGraph, vector_store: VectorMemoryStore):
        self.rdf_graph = rdf_graph
        self.vector_store = vector_store
    
    async def consolidate_session(self, session_id: str):
        """Background consolidation of conversation facts"""
        
        # Load session facts from SQLite
        session_facts = self._load_session_facts(session_id)
        
        # Add to RDF graph with metadata
        for fact in session_facts:
            self.rdf_graph.add_fact_with_metadata(
                fact['subject'], fact['relation'], fact['object'],
                confidence=fact['confidence'],
                timestamp=fact['timestamp'], 
                source=f"session_{session_id}"
            )
        
        # Run OWL inference
        self._run_owl_inference()
        
        # Validate with SHACL
        self._validate_shacl()
        
        # Export RDF-star dump
        self._export_rdf_dump(session_id)
    
    def _run_owl_inference(self):
        """Apply OWL reasoning rules"""
        # Example: If X drinks coffee and coffee is a beverage, 
        # then X consumes beverage
        inference_rules = """
        @prefix locat: <https://localcat.ai/> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        
        locat:coffee rdf:type locat:Beverage .
        
        { ?x locat:drinks locat:coffee }
        =>
        { ?x locat:consumes locat:beverage }
        """
        
        self.graph.update(inference_rules)
```

### Phase 3: Enhanced Retrieval Pipeline (Week 5-6)

#### Multi-Modal Retrieval
```python
# server/enhanced_retrieval.py
class EnhancedRetrieval:
    def __init__(self, vector_store: VectorMemoryStore, rdf_graph: RDFStarGraph):
        self.vector_store = vector_store
        self.rdf_graph = rdf_graph
        self.adjacency_store = MemoryStore()  # Existing LMDB
    
    async def retrieve_context(self, query: str, entities: List[str], 
                             max_bullets: int = 3) -> List[str]:
        """Enhanced retrieval with multiple strategies"""
        
        bullets = []
        
        # Strategy 1: Adjacency (fastest, existing)
        bullets.extend(await self._adjacency_retrieve(entities, max_bullets))
        
        # Strategy 2: Vector similarity (if needed)
        if len(bullets) < max_bullets:
            bullets.extend(await self._vector_retrieve(entities, max_bullets - len(bullets)))
        
        # Strategy 3: RDF reasoning (background, cached)
        if len(bullets) < max_bullets:
            bullets.extend(await self._rdf_retrieve(entities, max_bullets - len(bullets)))
        
        return bullets[:max_bullets]
    
    async def _vector_retrieve(self, entities: List[str], limit: int) -> List[str]:
        """Vector similarity retrieval"""
        bullets = []
        
        for entity in entities[:2]:  # Limit to avoid latency
            # Get entity vector
            entity_vector = await self.vector_store.get_entity_vector(entity)
            if entity_vector is None:
                continue
                
            # Find similar entities
            similar = await self.vector_store.find_similar_entities(entity_vector, limit=3)
            
            for sim_entity, similarity in similar:
                if similarity < 0.3:  # Similarity threshold
                    continue
                    
                # Get facts about similar entity
                facts = await self.adjacency_store.neighbors(sim_entity, "*")
                for dst, w, ts, pos, neg, status in facts[:1]:
                    if w > 0.5:  # Confidence threshold
                        bullets.append(f"• [similar] {sim_entity} relates to {dst}")
                        break
        
        return bullets
    
    async def _rdf_retrieve(self, entities: List[str], limit: int) -> List[str]:
        """RDF-star reasoning retrieval (cached results)"""
        bullets = []
        
        # Use cached RDF query results
        for entity in entities[:1]:
            cached_facts = await self._get_cached_rdf_facts(entity)
            for fact in cached_facts[:limit]:
                bullets.append(f"• [reasoned] {fact['subject']} {fact['relation']} {fact['object']}")
        
        return bullets
```

### Phase 4: Background Services & Export (Week 7-8)

#### Scheduled Consolidation
```python
# server/background_services.py
class BackgroundServices:
    def __init__(self):
        self.consolidator = GraphConsolidator()
        self.vector_processor = VectorProcessor()
    
    async def start_services(self):
        """Start background processing loops"""
        
        # Session consolidation (every 5 minutes)
        asyncio.create_task(self._session_consolidation_loop())
        
        # Vector index optimization (daily)
        asyncio.create_task(self._vector_optimization_loop())
        
        # RDF export (weekly)
        asyncio.create_task(self._rdf_export_loop())
    
    async def _session_consolidation_loop(self):
        """Consolidate completed sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                completed_sessions = self._find_completed_sessions()
                for session_id in completed_sessions:
                    await self.consolidator.consolidate_session(session_id)
                    
            except Exception as e:
                logger.error(f"Session consolidation error: {e}")
    
    async def _rdf_export_loop(self):
        """Weekly RDF-star export"""
        while True:
            try:
                await asyncio.sleep(604800)  # 1 week
                
                export_path = f"exports/knowledge_graph_{int(time.time())}.ttl"
                self.rdf_graph.export_turtle(export_path)
                
                logger.info(f"Exported RDF-star graph to {export_path}")
                
            except Exception as e:
                logger.error(f"RDF export error: {e}")
```

## 📈 Performance Targets

| Component | Target | Current | Method |
|-----------|--------|---------|--------|
| VectorLiteDB query | <50ms | - | SQLite vector indexing |
| Oxigraph SPARQL | <200ms | - | Cached query results |
| RDF export | <5000ms | - | Background processing |
| Storage overhead | <100MB | - | Compressed vectors + triples |
| Total cold path | <1000ms | - | Async processing |

## 🔬 Evaluation Metrics

### Accuracy Metrics
- **Retrieval Precision@3**: Target >0.85 (vs baseline adjacency)
- **RDF Reasoning Correctness**: Target >0.90 (manual validation)
- **Vector Similarity Quality**: Target >0.80 correlation with human judgment

### Performance Metrics
- **Cold Path Latency**: P99 <1000ms
- **Storage Growth**: <50MB/month for active users
- **Export Time**: <5 minutes for 1000 facts

### Quality Metrics
- **RDF-Star Compliance**: 100% valid triples
- **SPARQL Query Success**: >95% queries return results
- **Knowledge Portability**: Successful import into external tools

## 🚨 Risk Mitigation

### Technical Risks
1. **Storage Complexity**: Dual writes to SQLite + VectorLiteDB + Oxigraph
   - **Mitigation**: Abstract with unified MemoryStore interface
2. **RDF Learning Curve**: SPARQL complexity for team
   - **Mitigation**: Start with simple queries, build expertise gradually
3. **Vector Quality**: Poor embeddings hurt retrieval
   - **Mitigation**: Fine-tune on domain data, A/B test similarity thresholds

### Performance Risks
1. **Cold Path Delays**: Background processing impacting user experience
   - **Mitigation**: Strict timeouts, graceful degradation
2. **Storage Bloat**: RDF-star annotations increase size
   - **Mitigation**: Compression, selective annotation
3. **Query Latency**: SPARQL joins can be slow
   - **Mitigation**: Pre-compute common queries, use caching

## 🔄 Integration with GEPA

VectorLiteDB and Oxigraph can be optimized by GEPA:

```python
# GEPA can learn:
# - Optimal vector similarity thresholds per domain
# - Effective SPARQL query patterns
# - RDF reasoning rules that improve accuracy
# - Vector embedding dimensions for better retrieval

gepa_feedback = {
    'execution': {
        'vector_retrieval': vector_query_traces,
        'rdf_reasoning': sparql_query_traces,
        'ground_truth': user_corrected_retrieval
    },
    'metric': {
        'retrieval_precision': precision_score,
        'rdf_correctness': reasoning_accuracy,
        'latency_ms': end_to_end_time
    }
}
```

## 📚 References & Resources

### Libraries
- **VectorLiteDB**: https://github.com/vectorlitedb/vectorlitedb
- **Oxigraph**: https://github.com/oxigraph/oxigraph
- **RDFLib**: https://github.com/RDFLib/rdflib (Python RDF toolkit)
- **Sentence Transformers**: https://github.com/UKPLab/sentence-transformers

### Research Papers
- **RDF-Star**: https://w3c.github.io/rdf-star/cg-spec/
- **Vector Databases**: "Approximate Nearest Neighbor Search" surveys
- **Knowledge Graph Embedding**: TransE, DistMult, ComplEx papers

### Standards
- **RDF 1.1**: https://www.w3.org/TR/rdf11-primer/
- **SPARQL 1.1**: https://www.w3.org/TR/sparql11-query/
- **OWL 2**: https://www.w3.org/TR/owl2-primer/

## 🎯 Success Criteria

- [ ] VectorLiteDB enhances retrieval precision by 15%
- [ ] Oxigraph enables complex SPARQL queries
- [ ] RDF-star export works with external tools
- [ ] Cold path processing stays under 1000ms
- [ ] No impact on hot path (<200ms) performance
- [ ] GEPA can optimize vector and RDF parameters

## 📅 Timeline

**Week 1-2**: VectorLiteDB foundation and vector computation pipeline
**Week 3-4**: Oxigraph RDF-star graph and triple generation
**Week 5-6**: Enhanced retrieval with multi-modal strategies
**Week 7-8**: Background services, consolidation, and export
**Month 3**: GEPA optimization of thresholds and query patterns
**Month 4**: Production deployment with monitoring

---

*This integration completes LocalCat's memory system evolution: fast compiled RAM for voice → enhanced vector retrieval → rich semantic reasoning with RDF-star compliance, all while maintaining the speed that makes LocalCat unique.*