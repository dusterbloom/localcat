#!/usr/bin/env python3
"""
YAML UD Loader V2 - Advanced Conditional Pattern Support
Supports V8+ conditional emit patterns, guards, and complex rule structures
"""

import yaml
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class ConditionalEmit:
    """Advanced conditional emit with if/else logic"""
    condition: str
    subj: str
    pred: str
    obj: str
    triple_type: str = "default"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UDPatternV2:
    """Enhanced UD pattern with conditional support"""
    name: str
    priority: int
    description: str
    anchor_filter: Dict[str, Any]
    edges: List[Dict[str, Any]]
    guards: Dict[str, Any]
    emit: List[ConditionalEmit]
    confidence: float
    kind: str = "relation"
    helpers: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)

class ConditionalEvaluator:
    """Evaluates conditional expressions in emit patterns"""

    def __init__(self):
        self.operators = {
            'and': lambda a, b: a and b,
            'or': lambda a, b: a or b,
            'not': lambda a: not a,
            'in': lambda a, b: a in b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '>': lambda a, b: a > b,
            '<': lambda a, b: a < b,
            'contains': lambda a, b: b in str(a)
        }

    def evaluate(self, condition: str, variables: Dict[str, Any]) -> bool:
        """Evaluate conditional expression with variables"""
        try:
            # Handle simple variable checks
            if condition in variables:
                return bool(variables[condition])

            # Handle "var and var2" patterns
            if ' and ' in condition:
                parts = [p.strip() for p in condition.split(' and ')]
                return all(self._eval_simple(part, variables) for part in parts)

            # Handle "var or var2" patterns
            if ' or ' in condition:
                parts = [p.strip() for p in condition.split(' or ')]
                return any(self._eval_simple(part, variables) for part in parts)

            # Handle "not var" patterns
            if condition.startswith('not '):
                var = condition[4:].strip()
                return not self._eval_simple(var, variables)

            # Handle "var == 'value'" patterns
            if '==' in condition:
                left, right = [p.strip().strip('"\'') for p in condition.split('==')]
                return self._get_value(left, variables) == right

            # Handle "var in ['a', 'b', 'c']" patterns
            if ' in [' in condition:
                var, list_str = condition.split(' in ')
                var = var.strip()
                # Parse list - extract items between quotes
                items = re.findall(r"'([^']*)'", list_str)
                value = self._get_value(var, variables)
                return value in items

            # Handle "var.lemma" attribute access
            if '.' in condition:
                return self._eval_attribute_access(condition, variables)

            return self._eval_simple(condition, variables)

        except Exception as e:
            print(f"⚠️  Condition evaluation error: {condition} -> {e}")
            return False

    def _eval_simple(self, expr: str, variables: Dict[str, Any]) -> bool:
        """Evaluate simple expression"""
        expr = expr.strip()

        # Handle attribute access
        if '.' in expr:
            return self._eval_attribute_access(expr, variables)

        # Check if variable exists and is truthy
        if expr in variables:
            val = variables[expr]
            return val is not None and val != '' and val is not False

        return False

    def _eval_attribute_access(self, expr: str, variables: Dict[str, Any]) -> bool:
        """Handle object.attribute access patterns"""
        if '.' in expr:
            obj_name, attr = expr.split('.', 1)
            if obj_name in variables:
                obj = variables[obj_name]
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    return val is not None and val != ''
        return False

    def _get_value(self, expr: str, variables: Dict[str, Any]) -> Any:
        """Get value from expression, handling attributes"""
        if '.' in expr:
            obj_name, attr = expr.split('.', 1)
            if obj_name in variables:
                obj = variables[obj_name]
                if hasattr(obj, attr):
                    return getattr(obj, attr)
        elif expr in variables:
            return variables[expr]
        return None

class TextTemplateProcessor:
    """Processes template strings like {agent.text} with variables"""

    def process(self, template: str, variables: Dict[str, Any]) -> str:
        """Process template string with variable substitution"""
        result = template

        # Find all {var} or {var.attr} patterns
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, template)

        for match in matches:
            try:
                value = self._resolve_variable(match, variables)
                if value is not None:
                    result = result.replace('{' + match + '}', str(value))
                else:
                    result = result.replace('{' + match + '}', '')
            except:
                result = result.replace('{' + match + '}', '')

        return result.strip()

    def _resolve_variable(self, var_expr: str, variables: Dict[str, Any]) -> Any:
        """Resolve variable expression like 'agent.text' or 'agent.text or \'\''"""
        var_expr = var_expr.strip()

        # Handle "var.attr or 'default'" patterns
        if ' or ' in var_expr:
            main_expr, default = var_expr.split(' or ', 1)
            main_expr = main_expr.strip()
            default = default.strip().strip('"\'')

            value = self._resolve_simple_var(main_expr, variables)
            return value if value else default

        return self._resolve_simple_var(var_expr, variables)

    def _resolve_simple_var(self, var_expr: str, variables: Dict[str, Any]) -> Any:
        """Resolve simple variable like 'agent' or 'agent.text'"""
        if '.' in var_expr:
            obj_name, attr = var_expr.split('.', 1)
            if obj_name in variables:
                obj = variables[obj_name]
                if hasattr(obj, attr):
                    return getattr(obj, attr)
        elif var_expr in variables:
            return variables[var_expr]

        return None

class YAMLUDLoaderV2:
    """Enhanced YAML UD loader with conditional support"""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.meta = {}
        self.rules = []
        self.conditional_evaluator = ConditionalEvaluator()
        self.template_processor = TextTemplateProcessor()

    def load(self) -> List[UDPatternV2]:
        """Load and parse V2 YAML with conditional support"""
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        self.meta = data.get('meta', {})
        patterns_data = data.get('patterns', []) or data.get('rules', [])

        for pattern_data in patterns_data:
            pattern = self._compile_pattern_v2(pattern_data)
            if pattern:
                self.rules.append(pattern)

        # Sort by priority
        self.rules.sort(key=lambda x: x.priority, reverse=True)
        return self.rules

    def _compile_pattern_v2(self, pattern_data: Dict[str, Any]) -> Optional[UDPatternV2]:
        """Compile V2 pattern with conditional emit support"""
        try:
            name = pattern_data['name']
            priority = pattern_data.get('priority', 50)
            description = pattern_data.get('description', '')
            confidence = pattern_data.get('confidence', 0.95)

            pattern_dict = pattern_data['pattern']
            anchor_filter = pattern_dict.get('anchor', {})
            edges = pattern_dict.get('edges', [])

            guards = pattern_data.get('guards', {})
            emit_data = pattern_data.get('emit', [])

            # Parse conditional emit patterns
            conditional_emits = []
            for emit_item in emit_data:
                if isinstance(emit_item, dict):
                    # Check for conditional patterns
                    condition = emit_item.get('if', 'true')  # Default always true

                    conditional_emit = ConditionalEmit(
                        condition=condition,
                        subj=emit_item.get('subj', ''),
                        pred=emit_item.get('pred', ''),
                        obj=emit_item.get('obj', ''),
                        triple_type=emit_item.get('type', 'default'),
                        confidence=emit_item.get('confidence', confidence),
                        metadata={k: v for k, v in emit_item.items()
                                if k not in ['if', 'subj', 'pred', 'obj', 'type', 'confidence']}
                    )
                    conditional_emits.append(conditional_emit)
                else:
                    # Handle simple emit format for backward compatibility
                    conditional_emit = ConditionalEmit(
                        condition='true',
                        subj=str(emit_item.get('subj', '')),
                        pred=str(emit_item.get('pred', '')),
                        obj=str(emit_item.get('obj', '')),
                        confidence=confidence
                    )
                    conditional_emits.append(conditional_emit)

            return UDPatternV2(
                name=name,
                priority=priority,
                description=description,
                anchor_filter=anchor_filter,
                edges=edges,
                guards=guards,
                emit=conditional_emits,
                confidence=confidence,
                kind=pattern_data.get('kind', 'relation'),
                helpers=pattern_data.get('helpers', {}),
                tags=pattern_data.get('tags', {})
            )

        except Exception as e:
            print(f"⚠️  Pattern compilation error in {pattern_data.get('name', 'unknown')}: {e}")
            return None

class UDMatcherV2:
    """Enhanced UD matcher with conditional emit support"""

    def __init__(self, patterns: List[UDPatternV2], meta: Dict[str, Any]):
        self.patterns = patterns
        self.meta = meta
        self.conditional_evaluator = ConditionalEvaluator()
        self.template_processor = TextTemplateProcessor()
        print(f"✅ Loaded {len(patterns)} V2 UD patterns with conditional support")

    def extract_triples(self, doc) -> List[Tuple[str, str, str]]:
        """Extract triples using V2 conditional patterns"""
        all_triples = []

        for sent in doc.sents:
            sent_triples = self._extract_from_sentence(sent)
            all_triples.extend(sent_triples)

        return all_triples

    def _extract_from_sentence(self, sent) -> List[Tuple[str, str, str]]:
        """Extract triples from a single sentence using V2 patterns"""
        triples = []

        for pattern in self.patterns:
            pattern_matches = self._find_pattern_matches(sent, pattern)

            for match in pattern_matches:
                # Check guards
                if not self._check_guards_v2(match, pattern.guards, sent):
                    continue

                # Process conditional emits
                pattern_triples = self._process_conditional_emits(match, pattern.emit, sent)
                triples.extend(pattern_triples)

        return triples

    def _find_pattern_matches(self, sent, pattern: UDPatternV2) -> List[Dict[str, Any]]:
        """Find all matches for a pattern in sentence"""
        matches = []

        # Find anchor matches
        for token in sent:
            if self._matches_anchor_v2(token, pattern.anchor_filter):
                match = {'anchor': token}

                # Try to match all edges
                if self._match_edges_v2(match, pattern.edges, sent):
                    matches.append(match)

        return matches

    def _matches_anchor_v2(self, token, anchor_filter: Dict[str, Any]) -> bool:
        """Enhanced anchor matching with V2 support"""
        for key, value in anchor_filter.items():
            if key == 'pos':
                if not self._matches_pos_pattern(token, value):
                    return False
            elif key == 'lemma':
                if not self._matches_lemma_pattern(token, value):
                    return False
            elif key == 'dep':
                if not self._matches_dep_pattern(token, value):
                    return False
            elif key == 'tag':
                if not self._matches_tag_pattern(token, value):
                    return False
            elif key == 'lemma_not':
                if isinstance(value, list):
                    if token.lemma_.lower() in [v.lower() for v in value]:
                        return False
                else:
                    if token.lemma_.lower() == value.lower():
                        return False

        return True

    def _matches_pos_pattern(self, token, pattern: str) -> bool:
        """Match POS pattern with OR support"""
        if '|' in pattern:
            return token.pos_ in pattern.split('|')
        return token.pos_ == pattern

    def _matches_lemma_pattern(self, token, pattern: str) -> bool:
        """Match lemma pattern with OR support"""
        if '|' in pattern:
            lemmas = [l.strip() for l in pattern.split('|')]
            return token.lemma_.lower() in [l.lower() for l in lemmas]
        return token.lemma_.lower() == pattern.lower()

    def _matches_dep_pattern(self, token, pattern: str) -> bool:
        """Match dependency pattern"""
        if '|' in pattern:
            return token.dep_ in pattern.split('|')
        return token.dep_ == pattern

    def _matches_tag_pattern(self, token, pattern: str) -> bool:
        """Match tag pattern with OR support"""
        if '|' in pattern:
            return token.tag_ in pattern.split('|')
        return token.tag_ == pattern

    def _match_edges_v2(self, match: Dict[str, Any], edges: List[Dict[str, Any]], sent) -> bool:
        """Enhanced edge matching with V2 support"""
        for edge in edges:
            if not self._match_single_edge_v2(match, edge, sent):
                return False
        return True

    def _match_single_edge_v2(self, match: Dict[str, Any], edge: Dict[str, Any], sent) -> bool:
        """Match a single edge with enhanced V2 support"""
        from_token = match[edge['from']]
        rel_pattern = edge['rel']
        edge_name = edge['as']

        # Handle optional edges
        optional = edge.get('required', True) == False or edge_name.endswith('?')
        edge_key = edge_name.rstrip('?')

        # Find matching dependencies
        candidates = []

        # Handle different relationship patterns
        if rel_pattern.startswith('^'):
            # Child relationships
            rel_clean = rel_pattern[1:]
            for child in from_token.children:
                if self._matches_rel_pattern(child.dep_, rel_clean):
                    if self._matches_edge_constraints(child, edge):
                        candidates.append(child)
        else:
            # Parent/sibling relationships
            for token in sent:
                if token.head == from_token and self._matches_rel_pattern(token.dep_, rel_pattern):
                    if self._matches_edge_constraints(token, edge):
                        candidates.append(token)

        if not candidates:
            if optional:
                return True  # Optional edge not found, continue
            else:
                return False  # Required edge not found, fail

        # Use first candidate (could be enhanced to handle multiple)
        match[edge_key] = candidates[0]
        return True

    def _matches_rel_pattern(self, dep: str, pattern: str) -> bool:
        """Match dependency relation pattern"""
        if '|' in pattern:
            return dep in pattern.split('|')
        return dep == pattern

    def _matches_edge_constraints(self, token, edge: Dict[str, Any]) -> bool:
        """Check additional edge constraints"""
        # POS constraint
        if 'pos' in edge:
            if not self._matches_pos_pattern(token, edge['pos']):
                return False

        # Lemma constraint
        if 'lemma' in edge:
            if not self._matches_lemma_pattern(token, edge['lemma']):
                return False

        # Case constraint (for prepositions)
        if 'case' in edge:
            case_pattern = edge['case']
            if '|' in case_pattern:
                cases = case_pattern.split('|')
                if not any(case.lower() in token.text.lower() for case in cases):
                    return False
            else:
                if case_pattern.lower() not in token.text.lower():
                    return False

        return True

    def _check_guards_v2(self, match: Dict[str, Any], guards: Dict[str, Any], sent) -> bool:
        """Enhanced guard checking with V2 support"""
        for guard_name, guard_value in guards.items():
            if not self._evaluate_guard_v2(guard_name, guard_value, match, sent):
                return False
        return True

    def _evaluate_guard_v2(self, guard_name: str, guard_value: Any, match: Dict[str, Any], sent) -> bool:
        """Evaluate a single guard with V2 enhanced logic"""
        try:
            # Handle exclude patterns
            if guard_name == 'exclude_lemma' and isinstance(guard_value, list):
                anchor = match.get('anchor')
                if anchor and anchor.lemma_.lower() in [v.lower() for v in guard_value]:
                    return False

            # Handle lemma inclusion patterns
            elif guard_name.endswith('_lemma_in') and isinstance(guard_value, list):
                var_name = guard_name.replace('_lemma_in', '')
                if var_name in match:
                    token = match[var_name]
                    if token.lemma_.lower() not in [v.lower() for v in guard_value]:
                        return False

            # Handle position constraints
            elif guard_name.endswith('_pos') and isinstance(guard_value, str):
                var_name = guard_name.replace('_pos', '')
                if var_name in match:
                    token = match[var_name]
                    if token.pos_ != guard_value:
                        return False

            # Handle custom logical guards
            elif isinstance(guard_value, bool):
                return guard_value  # Static true/false

            # Handle complex conditions
            elif isinstance(guard_value, str):
                return self.conditional_evaluator.evaluate(guard_value, match)

            return True

        except Exception as e:
            print(f"⚠️  Guard evaluation error {guard_name}: {e}")
            return False

    def _process_conditional_emits(self, match: Dict[str, Any], emits: List[ConditionalEmit], sent) -> List[Tuple[str, str, str]]:
        """Process conditional emit patterns to generate triples"""
        triples = []

        for emit in emits:
            # Evaluate condition
            if self.conditional_evaluator.evaluate(emit.condition, match):
                # Process template strings
                subj = self.template_processor.process(emit.subj, match)
                pred = self.template_processor.process(emit.pred, match)
                obj = self.template_processor.process(emit.obj, match)

                # Validate triple quality
                if subj and pred and self._is_valid_triple(subj, pred, obj):
                    triples.append((subj, pred, obj))

        return triples

    def _is_valid_triple(self, subj: str, pred: str, obj: str) -> bool:
        """Validate triple quality"""
        # Basic length check
        if len(subj.strip()) < 1 or len(pred.strip()) < 1:
            return False

        # No empty predicates
        if pred.strip() in ['', 'unknown', 'generic']:
            return False

        return True

class YAMLUDExtractorV2:
    """Enhanced V2 extractor with conditional pattern support"""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.matcher = None
        self._load_rules_v2()

    def _load_rules_v2(self):
        """Load V2 rules with conditional support"""
        loader = YAMLUDLoaderV2(self.yaml_path)
        patterns = loader.load()
        self.matcher = UDMatcherV2(patterns, loader.meta)
        print(f"✅ Loaded {len(patterns)} V2 UD→SRL rules from {self.yaml_path}")

    def extract_triples(self, doc) -> List[Tuple[str, str, str]]:
        """Extract semantic triples using V2 conditional patterns"""
        if not self.matcher:
            return []
        return self.matcher.extract_triples(doc)

# Test function for V2
def test_yaml_ud_extractor_v2():
    """Test V2 extractor with conditional patterns"""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Please install: python -m spacy download en_core_web_sm")
        return

    print("🔥 TESTING YAML UD EXTRACTOR V2 (Conditional Support)")
    print("=" * 60)

    extractor = YAMLUDExtractorV2("ULTRAGROK_V8.yaml")

    test_sentences = [
        "John works at Google.",
        "My name is Alex Thompson.",
        "Mary lives in Paris.",
        "The CEO announced profits.",
        "John gave Mary a book."
    ]

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📝 Test {i}: {sentence}")
        print("-" * 40)

        doc = nlp(sentence)
        triples = extractor.extract_triples(doc)

        print(f"🔗 {len(triples)} triples extracted:")
        for triple in triples:
            print(f"   • {triple}")

if __name__ == "__main__":
    test_yaml_ud_extractor_v2()