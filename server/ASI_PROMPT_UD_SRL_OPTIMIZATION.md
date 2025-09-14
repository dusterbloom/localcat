# ASI Prompt: UD→SRL Rule Optimization for Production Semantic Extraction

## Context & Mission
We're building a production-ready **multilingual semantic role labeling system** that converts Universal Dependencies parse trees into high-quality semantic triples for knowledge graphs. The goal is **<500ms extraction with universal language support** while maintaining semantic quality comparable to English-only AllenNLP systems.

## What We Have
A YAML-based rule system (`fastlane_rules.ud.yaml`) with 150-line Python loader that:
- Processes UD parse trees from spaCy/Stanza
- Matches declarative patterns (anchor + edge constraints)
- Emits semantic triples like `("you", "has_name", "Alex Thompson")`
- Handles complex constructions: ditransitives, passives, copulas, relatives

## Current Coverage Analysis
**✅ Working Well:**
- Basic SVO: `("John", "gave", "Mary")`
- Copulas: `("name", "be", "Alex")` from "My name is Alex"
- Verb+prep: `("live", "live_in", "Paris")`
- Passives: `("agent", "action", "patient")` normalization

**🔄 Needs Enhancement:**
- **Coordination**: `"John and Mary went"` → extract for both subjects
- **Quantifiers**: `"all students"`, `"every person"`, `"some books"`
- **Complex nominals**: `"CEO of Microsoft"`, `"teacher at school"`
- **Temporal chains**: `"after graduating, he worked"`
- **Modals/aspectuals**: `"will do"`, `"has done"`, `"might go"`
- **Comparative constructions**: `"better than"`, `"as good as"`
- **Clause embedding**: `"I think that he knows"`, `"she said he left"`

## ASI Optimization Challenge

**Your mission:** Analyze and enhance our UD→SRL rule system to achieve **90%+ semantic coverage** on real-world text while maintaining **<500ms** performance.

### Key Focus Areas:

1. **Rule Completeness**
   - Identify missing UD construction patterns that occur frequently in practice
   - Add rules for coordination propagation (`conj` → replicate triples)
   - Handle complex predicate structures (auxiliaries, modals, aspectuals)
   - Cover noun phrase internal structure (`compound`, `flat`, `nmod` chains)

2. **Quality Improvements**
   - Refine predicate naming conventions for consistency
   - Add entity coreference/merging hints for pronouns
   - Improve temporal and aspectual marking
   - Handle negation propagation properly

3. **Cross-linguistic Robustness**
   - Ensure patterns work across major language families
   - Add language-specific lemma lists where needed (minimal)
   - Handle different UD annotation conventions gracefully

4. **Edge Case Coverage**
   - Elliptical constructions and gapping
   - Questions and imperatives
   - Idiomatic expressions
   - Coordination scope ambiguities

### Specific Requests:

**A) Pattern Enhancement**
```yaml
# Add ~10-15 new high-impact rules covering:
- Coordination propagation (conj + cc)
- Complex nominal predicates ("CEO of", "teacher at")
- Modal/aspectual verb chains ("will be doing", "has been")
- Clause embedding with proper scoping
- Quantifier handling ("all X do Y" → "X do Y" + universal quantification)
```

**B) Template System Expansion**
```yaml
# Enhance template variables:
- {coord_subjects} - expand conjunctions automatically
- {modal_aspect_pred} - normalize modal+aspect+main verb
- {entity_canonical} - merge name components (flat:name, compound)
- {negation_marker} - propagate neg dependency
- {temporal_anchor} - extract temporal expressions
```

**C) Evaluation Framework**
Create test cases covering:
- 20 diverse sentence types across constructions
- Expected output triples for each
- Performance benchmarks (<500ms target)
- Cross-language validation (EN/ES/IT/DE/FR)

### Output Format:
Please provide:
1. **Enhanced YAML file** with 15-20 additional high-quality rules
2. **Analysis report** explaining optimization rationale
3. **Test suite** with challenging examples and expected outputs
4. **Performance predictions** and potential bottlenecks

### Success Metrics:
- **Coverage**: Extract meaningful triples from 90%+ of sentences
- **Quality**: Semantic relations suitable for knowledge graphs
- **Speed**: <500ms processing time per document
- **Universality**: Works across major languages with UD parsing

### Example Input/Output:
```
Input: "After John and Mary graduated, they both worked at Google."
Expected Output:
- ("John", "graduated", "")
- ("Mary", "graduated", "")
- ("John", "worked_at", "Google")
- ("Mary", "worked_at", "Google")
- ("graduated", "before", "worked")  # temporal
```

**Your expertise in linguistic patterns, rule optimization, and cross-language NLP will be invaluable for creating a production-ready semantic extraction system that surpasses existing multilingual SRL approaches.**