#!/usr/bin/env python3
"""
YAML-based Universal Dependencies to SRL Rule Loader
==================================================

Compiles fastlane_rules.ud.yaml into matchers for semantic role labeling.
Handles complex UD patterns with template variables and multiword constructions.
"""

import yaml
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class UDPattern:
    """Compiled UD pattern matcher"""
    name: str
    kind: str
    priority: int
    anchor_filter: Dict[str, Any]
    edges: List[Dict[str, Any]]
    emit: List[Dict[str, str]]
    guards: Dict[str, Any]
    helpers: Dict[str, bool]
    tags: Dict[str, str]

class YAMLUDLoader:
    """Loads and compiles YAML UD rules into executable patterns"""

    def __init__(self, yaml_path: str):
        self.yaml_path = Path(yaml_path)
        self.rules: List[UDPattern] = []
        self.meta: Dict[str, Any] = {}

    def load(self) -> List[UDPattern]:
        """Load and compile YAML rules"""
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        self.meta = data.get('meta', {})
        rules_data = data.get('rules', [])

        for rule_data in rules_data:
            pattern = self._compile_rule(rule_data)
            if pattern:
                self.rules.append(pattern)

        # Sort by priority (higher first)
        self.rules.sort(key=lambda x: x.priority, reverse=True)
        return self.rules

    def _compile_rule(self, rule_data: Dict[str, Any]) -> Optional[UDPattern]:
        """Compile a single rule into a pattern matcher"""
        try:
            name = rule_data['name']
            kind = rule_data.get('kind', 'relation')
            priority = rule_data.get('priority', 50)

            pattern = rule_data['pattern']
            anchor_filter = pattern['anchor']
            edges = pattern.get('edges', [])

            emit = rule_data.get('emit', [])
            guards = rule_data.get('guards', {})
            helpers = rule_data.get('helpers', {})
            tags = rule_data.get('set', {}) if kind == 'tag' else {}

            return UDPattern(
                name=name,
                kind=kind,
                priority=priority,
                anchor_filter=anchor_filter,
                edges=edges,
                emit=emit,
                guards=guards,
                helpers=helpers,
                tags=tags
            )
        except Exception as e:
            print(f"Error compiling rule {rule_data.get('name', 'unknown')}: {e}")
            return None

class UDMatcher:
    """Matches UD patterns against dependency trees and extracts semantic triples"""

    def __init__(self, patterns: List[UDPattern], meta: Dict[str, Any]):
        self.patterns = patterns
        self.meta = meta
        self.joiner = meta.get('joiner', {})
        self.engine = meta.get('engine', {})

    def extract_triples(self, doc) -> List[Tuple[str, str, str]]:
        """Extract semantic triples from spaCy doc using compiled patterns"""
        triples = []
        context = {'doc': doc}

        # Process each token as potential anchor
        for token in doc:
            for pattern in self.patterns:
                if self._matches_anchor(token, pattern.anchor_filter):
                    matches = self._match_pattern(token, pattern, context)
                    if matches:
                        for match in matches:
                            if pattern.kind == 'relation':
                                triple = self._emit_triple(match, pattern, context)
                                if triple:
                                    triples.append(triple)
                            elif pattern.kind == 'tag':
                                self._apply_tags(match, pattern, context)

        return self._deduplicate_triples(triples)

    def _matches_anchor(self, token, anchor_filter: Dict[str, Any]) -> bool:
        """Check if token matches anchor filter"""
        for key, value in anchor_filter.items():
            if key == 'pos':
                if not self._match_pos(token, value):
                    return False
            elif key == 'lemma':
                # Support OR pattern: "be|is|are|was|were"
                lemma_options = [v.strip() for v in value.split('|')]
                if token.lemma_.lower() not in [l.lower() for l in lemma_options]:
                    return False
            elif key == 'rel':
                if not re.match(value, token.dep_):
                    return False
        return True

    def _match_pos(self, token, pos_pattern: str) -> bool:
        """Match POS pattern (supports | for OR)"""
        pos_options = [p.strip() for p in pos_pattern.split('|')]
        return token.pos_ in pos_options

    def _match_pattern(self, anchor, pattern: UDPattern, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match edge patterns from anchor token"""
        matches = [{'anchor': anchor}]

        for edge in pattern.edges:
            new_matches = []
            for match in matches:
                edge_matches = self._match_edge(match, edge, context)
                new_matches.extend(edge_matches)
            matches = new_matches

            if not matches:
                break

        # Apply guards
        if matches and pattern.guards:
            matches = [m for m in matches if self._check_guards(m, pattern.guards, context)]

        return matches

    def _match_edge(self, match: Dict[str, Any], edge: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match a single edge pattern"""
        from_token = match[edge['from']]
        rel_pattern = edge['rel']
        edge_name = edge['as']
        optional = edge_name.endswith('?')
        edge_key = edge_name.rstrip('?')

        # Find matching children/dependencies
        candidates = []
        for child in from_token.children:
            if re.match(rel_pattern, child.dep_):
                if 'pos' in edge:
                    if not self._match_pos(child, edge['pos']):
                        continue
                candidates.append(child)

        if not candidates:
            if optional:
                return [match]  # Continue without this edge
            else:
                return []  # Required edge not found

        # Return all combinations
        results = []
        for candidate in candidates:
            new_match = match.copy()
            new_match[edge_key] = candidate
            results.append(new_match)

        return results

    def _check_guards(self, match: Dict[str, Any], guards: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check guard conditions"""
        for guard, value in guards.items():
            if guard == 'verb_lemma_in':
                anchor = match['anchor']
                if anchor.lemma_.lower() not in [v.lower() for v in value]:
                    return False
            elif guard == 'require_case_lemma_in':
                if 'case' in match:
                    case_token = match['case']
                    if case_token.lemma_.lower() not in [c.lower() for c in value]:
                        return False
                else:
                    return False
            elif guard == 'drop_if_obj_missing':
                if not match.get('obj') and value:
                    return False
            elif guard.endswith('_lemma_in'):
                # Generic lemma check for any matched variable (e.g., name_noun_lemma_in, poss_pron_lemma_in)
                var_name = guard.replace('_lemma_in', '')
                if var_name in match:
                    token = match[var_name]
                    if token.lemma_.lower() not in [v.lower() for v in value]:
                        return False
                else:
                    return False
            elif guard.endswith('_pos'):
                # POS check for any matched variable (e.g., adj_pos)
                var_name = guard.replace('_pos', '')
                if var_name in match:
                    token = match[var_name]
                    if token.pos_ != value:
                        return False
                else:
                    return False
        return True

    def _emit_triple(self, match: Dict[str, Any], pattern: UDPattern, context: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
        """Emit semantic triple from pattern match"""
        if not pattern.emit:
            return None

        # Use first emit template (could extend for multiple)
        emit_template = pattern.emit[0]

        try:
            subj = self._resolve_template(emit_template.get('subj', ''), match, context)
            pred = self._resolve_template(emit_template.get('pred', ''), match, context)
            obj = self._resolve_template(emit_template.get('obj', ''), match, context)

            # Clean and validate
            subj = self._clean_text(subj)
            pred = self._clean_text(pred)
            obj = self._clean_text(obj)

            if subj and pred:  # obj can be empty for some relations
                return (subj, pred, obj or "")
        except Exception as e:
            print(f"Error emitting triple for {pattern.name}: {e}")

        return None

    def _resolve_template(self, template: str, match: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Resolve template variables like {subj.text}, {verb_lemma}"""
        if not template:
            return ""

        # Handle complex templates
        result = template

        # Simple token references
        for key, token in match.items():
            if hasattr(token, 'text'):
                result = result.replace(f"{{{key}.text}}", token.text)
                result = result.replace(f"{{{key}.lemma}}", token.lemma_)

        # Special variables
        anchor = match.get('anchor')
        if anchor:
            result = result.replace('{verb_lemma}', anchor.lemma_ if anchor.pos_ == 'VERB' else anchor.lemma_)

        # Complex head references
        if '{verb_head.lemma}' in result:
            # Find verb head for coordination patterns
            anchor_token = match.get('anchor')
            if anchor_token and anchor_token.head and anchor_token.head.pos_ == 'VERB':
                result = result.replace('{verb_head.lemma}', anchor_token.head.lemma_)
            else:
                result = result.replace('{verb_head.lemma}', 'unknown_verb')

        # Object references
        if '{verb_obj.text}' in result:
            anchor_token = match.get('anchor')
            if anchor_token and anchor_token.head:
                verb = anchor_token.head
                obj_token = None
                for child in verb.children:
                    if child.dep_ in ['obj', 'dobj']:
                        obj_token = child
                        break
                if obj_token:
                    result = result.replace('{verb_obj.text}', obj_token.text)
                else:
                    result = result.replace('{verb_obj.text}', '')
            else:
                result = result.replace('{verb_obj.text}', '')

        # Embedded clause handling
        if '{embedded.subtree}' in result:
            embedded = match.get('embedded')
            if embedded:
                # Get the text span of the embedded clause
                subtree_text = embedded.text
                result = result.replace('{embedded.subtree}', subtree_text)
            else:
                result = result.replace('{embedded.subtree}', '')

        # General subtree handling for any match variable
        import re
        subtree_pattern = r'\{(\w+)\.subtree\}'
        for subtree_match in re.finditer(subtree_pattern, result):
            var_name = subtree_match.group(1)
            token = match.get(var_name)
            if token:
                # Get subtree text including all dependent tokens
                subtree_tokens = list(token.subtree)
                subtree_text = ' '.join([t.text for t in subtree_tokens])
                result = result.replace(subtree_match.group(0), subtree_text)
            else:
                result = result.replace(subtree_match.group(0), '')

        # Prep suffix
        prep_suffix = ""
        if 'case' in match:
            case_token = match['case']
            prep_text = case_token.lemma_.lower()

            # Handle multiword prepositions via fixed chain
            if self.engine.get('read_mwadp', False):
                fixed_parts = [child.lemma_ for child in case_token.children if child.dep_ == 'fixed']
                if fixed_parts:
                    prep_text = f"{prep_text}_{'_'.join(fixed_parts)}"

            prep_suffix = f"_{prep_text}"

        result = result.replace('{prep_suffix}', prep_suffix)

        # Particle suffix
        prt_suffix = ""
        if 'prt' in match:
            prt_token = match['prt']
            prt_suffix = f"_{prt_token.lemma_}"
        result = result.replace('{prt_suffix}', prt_suffix)

        # Helper functions
        if '{obj_or_dep}' in result:
            obj_text = match.get('obj', match.get('dep', '')).text if match.get('obj') or match.get('dep') else ""
            result = result.replace('{obj_or_dep}', obj_text)

        return result

    def _apply_tags(self, match: Dict[str, Any], pattern: UDPattern, context: Dict[str, Any]):
        """Apply tags to tokens (for later processing)"""
        anchor = match['anchor']
        for tag_key, tag_value in pattern.tags.items():
            resolved_value = self._resolve_template(tag_value, match, context)
            # Store tag on token (extend spaCy token with custom attributes)
            if not hasattr(anchor, '_ud_tags'):
                anchor._ud_tags = {}
            anchor._ud_tags[tag_key] = resolved_value

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        return text.strip().replace('\n', ' ').replace('\t', ' ')

    def _deduplicate_triples(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Remove duplicate triples while preserving order"""
        seen = set()
        result = []
        for triple in triples:
            if triple not in seen:
                seen.add(triple)
                result.append(triple)
        return result

# Integration wrapper
class YAMLUDExtractor:
    """Production wrapper for YAML-based UD extraction"""

    def __init__(self, yaml_path: str = "fastlane_rules.ud.yaml"):
        self.yaml_path = yaml_path
        self.matcher = None
        self._load_rules()

    def _load_rules(self):
        """Load and compile rules"""
        loader = YAMLUDLoader(self.yaml_path)
        patterns = loader.load()
        self.matcher = UDMatcher(patterns, loader.meta)
        print(f"✅ Loaded {len(patterns)} UD→SRL rules from {self.yaml_path}")

    def extract_triples(self, doc) -> List[Tuple[str, str, str]]:
        """Extract semantic triples from spaCy doc"""
        if not self.matcher:
            return []
        return self.matcher.extract_triples(doc)

# Test function
def test_yaml_ud_extractor():
    """Test the YAML UD extractor"""
    import spacy

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Please install: python -m spacy download en_core_web_sm")
        return

    # Initialize extractor
    extractor = YAMLUDExtractor("fastlane_rules.ud.yaml")

    test_sentences = [
        "John gave Mary a book yesterday.",
        "My name is Alex Thompson.",
        "The company announced profits.",
        "Maria bought a car in Madrid last week.",
    ]

    print("🚀 Testing YAML-based UD→SRL Extractor")
    print("=" * 50)

    for sentence in test_sentences:
        print(f"\n📝 Sentence: {sentence}")

        doc = nlp(sentence)
        triples = extractor.extract_triples(doc)

        print(f"🔗 Extracted {len(triples)} triples:")
        for triple in triples:
            print(f"   • {triple}")
        print("-" * 30)

if __name__ == "__main__":
    test_yaml_ud_extractor()