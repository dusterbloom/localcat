#!/usr/bin/env python3
"""
Enhanced bullet point formatter for HotMem that generates rich, detailed summaries.
Handles a wide variety of relations and creates more informative bullet points.
"""

from typing import Optional, Dict, Set
import re


class EnhancedBulletFormatter:
    """
    Generates detailed, natural language bullet points from extracted triples.
    Handles complex relations and provides contextual information.
    """
    
    # Relation templates for richer bullet generation
    RELATION_TEMPLATES = {
        # Discovery/Creation relations
        "founded": "{s} founded {d}",
        "co-founded": "{s} co-founded {d}",
        "established": "{s} established {d}",
        "created": "{s} created {d}",
        "invented": "{s} invented {d}",
        "discovered": "{s} discovered {d}",
        "developed": "{s} developed {d}",
        "designed": "{s} designed {d}",
        "built": "{s} built {d}",
        "launched": "{s} launched {d}",
        
        # Professional relations
        "works_at": "{s} works at {d}",
        "worked_at": "{s} worked at {d}",
        "employed_by": "{s} is employed by {d}",
        "teaches_at": "{s} teaches at {d}",
        "taught_at": "{s} taught at {d}",
        "studied_at": "{s} studied at {d}",
        "graduated_from": "{s} graduated from {d}",
        "manages": "{s} manages {d}",
        "leads": "{s} leads {d}",
        "directs": "{s} directs {d}",
        
        # Location relations
        "lives_in": "{s} lives in {d}",
        "lived_in": "{s} lived in {d}",
        "resides_in": "{s} resides in {d}",
        "born_in": "{s} was born in {d}",
        "moved_from": "{s} moved from {d}",
        "moved_to": "{s} moved to {d}",
        "visited": "{s} visited {d}",
        "traveled_to": "{s} traveled to {d}",
        
        # Personal relations
        "married_to": "{s} is married to {d}",
        "engaged_to": "{s} is engaged to {d}",
        "parent_of": "{s} is the parent of {d}",
        "child_of": "{s} is the child of {d}",
        "sibling_of": "{s} is a sibling of {d}",
        "friend_of": "{s} is a friend of {d}",
        "colleague_of": "{s} is a colleague of {d}",
        "partner_of": "{s} is a partner of {d}",
        
        # Ownership/Possession
        "owns": "{s} owns {d}",
        "has": "{s} have {d}",  # Will be conjugated properly
        "possesses": "{s} possesses {d}",
        "acquired": "{s} acquired {d}",
        "bought": "{s} bought {d}",
        "sold": "{s} sold {d}",
        
        # Achievement relations
        "won": "{s} won {d}",
        "achieved": "{s} achieved {d}",
        "earned": "{s} earned {d}",
        "received": "{s} received {d}",
        "awarded": "{s} was awarded {d}",
        
        # Educational relations
        "studied": "{s} studied {d}",
        "researched": "{s} researched {d}",
        "specializes_in": "{s} specializes in {d}",
        "expert_in": "{s} is an expert in {d}",
        "published": "{s} published {d}",
        "wrote": "{s} wrote {d}",
        "authored": "{s} authored {d}",
        
        # Temporal relations
        "started": "{s} started {d}",
        "ended": "{s} ended {d}",
        "began": "{s} began {d}",
        "completed": "{s} completed {d}",
        "finished": "{s} finished {d}",
        
        # Identity relations
        "name": "{s}'s name is {d}",
        "also_known_as": "{s} is also known as {d}",
        "nicknamed": "{s} is nicknamed {d}",
        "called": "{s} is called {d}",
        
        # Attribute relations
        "age": "{s} is {d} years old",
        "height": "{s} is {d} tall",
        "weight": "{s} weighs {d}",
        "color": "{s}'s color is {d}",
        "favorite": "{s}'s favorite is {d}",
        "likes": "{s} likes {d}",
        "dislikes": "{s} dislikes {d}",
        "prefers": "{s} prefers {d}",
        
        # Generic relations
        "name": "{s}'s name is {d}",
        "is": "{s} is {d}",
        "was": "{s} was {d}",
        "became": "{s} became {d}",
        "represents": "{s} represents {d}",
        "serves": "{s} serves {d}",
        "supports": "{s} supports {d}",
        "uses": "{s} uses {d}",
        "produces": "{s} produces {d}",
        "provides": "{s} provides {d}",
        "offers": "{s} offers {d}",
        "delivers": "{s} delivers {d}",
    }
    
    # Special handling for complex subjects
    SUBJECT_ENHANCEMENTS = {
        "you": "You",
        "your": "Your",
        "brother": "Your brother",
        "sister": "Your sister",
        "mother": "Your mother",
        "father": "Your father",
        "son": "Your son",
        "daughter": "Your daughter",
        "friend": "Your friend",
        "colleague": "Your colleague",
        "boss": "Your boss",
        "teacher": "Your teacher",
        "student": "Your student",
        "pet": "Your pet",
        "dog": "Your dog",
        "cat": "Your cat",
    }
    
    def format_bullet(self, subject: str, relation: str, obj: str, 
                     include_context: bool = True) -> str:
        """
        Format a triple into a rich, detailed bullet point.
        
        Args:
            subject: The subject of the triple
            relation: The relation/predicate
            obj: The object of the triple
            include_context: Whether to add contextual information
            
        Returns:
            A formatted bullet point string
        """
        # Clean and normalize inputs
        s = self._clean_entity(str(subject) if subject is not None else "")
        r = self._clean_relation(str(relation) if relation is not None else "")
        d = self._clean_entity(str(obj) if obj is not None else "")
        
        # Enhance subject if needed
        s_enhanced = self.SUBJECT_ENHANCEMENTS.get(s.lower(), s)
        
        # Handle special cases first
        bullet = self._handle_special_cases(s, r, d)
        if bullet:
            return bullet
        
        # Try template matching
        if r in self.RELATION_TEMPLATES:
            template = self.RELATION_TEMPLATES[r]
            # Special handling for "has" relation - add article if needed
            if r == "has":
                d_formatted = self._add_article_if_needed(d)
                bullet = f"{s_enhanced} {self._conjugate_verb('have', s)} {d_formatted}"
            else:
                bullet = template.format(s=s_enhanced, d=d)
        # Handle verb relations (v:xyz format)
        elif r.startswith("v:"):
            verb = r[2:]
            bullet = f"{s_enhanced} {self._conjugate_verb(verb, s)} {d}"
        # Handle compound relations (verb_prep format)
        elif "_" in r:
            parts = r.split("_")
            if len(parts) == 2 and parts[1] in ["in", "at", "to", "from", "with", "for", "on", "by"]:
                verb, prep = parts
                bullet = f"{s_enhanced} {self._conjugate_verb(verb, s)} {prep} {d}"
            else:
                # Generic compound relation
                relation_phrase = r.replace("_", " ")
                bullet = f"{s_enhanced} {relation_phrase} {d}"
        # Handle simple verb relations
        elif self._is_verb(r):
            bullet = f"{s_enhanced} {self._conjugate_verb(r, s)} {d}"
        # Default fallback
        else:
            bullet = f"{s_enhanced} {r} {d}"
        
        # Add contextual enhancements
        if include_context:
            bullet = self._add_contextual_info(bullet, s, r, d)
        
        # Ensure proper formatting
        bullet = self._finalize_bullet(bullet)
        
        return bullet
    
    def _clean_entity(self, entity: str) -> str:
        """Clean and normalize entity text."""
        # Remove extra whitespace
        entity = " ".join(entity.split())
        # Remove quotes if present
        entity = entity.strip('"\'')
        return entity
    
    def _add_article_if_needed(self, noun: str) -> str:
        """Add appropriate article to a noun if needed."""
        # Don't add article if already has one
        if noun.lower().startswith(("a ", "an ", "the ", "my ", "your ", "his ", "her ", "their ", "our ")):
            return noun
        
        # Don't add article to proper nouns or names
        if noun and noun[0].isupper():
            return noun
        
        # Don't add article to pronouns or quantifiers
        if noun.lower() in ["someone", "something", "someone's", "everyone", "everything", "nothing", "nobody"]:
            return noun
        
        # Don't add article to plural or uncountable nouns (simple heuristic)
        if noun.endswith("s") and not noun.endswith("ss"):
            return noun
        
        # Add "a" or "an" based on first letter (simplified)
        first_letter = noun[0].lower() if noun else ""
        if first_letter in "aeiou":
            return f"an {noun}"
        else:
            return f"a {noun}"
    
    def _clean_relation(self, relation: str) -> str:
        """Clean and normalize relation text."""
        # Handle various relation formats
        relation = relation.lower().strip()
        # Remove common prefixes
        if relation.startswith("rel:"):
            relation = relation[4:]
        return relation
    
    def _handle_special_cases(self, s: str, r: str, d: str) -> Optional[str]:
        """Handle special formatting cases."""
        s_lower = s.lower()
        
        # Handle "you" + "name" special case to avoid "You's name"
        if s_lower == "you" and r == "name":
            return f"• Your name is {d}"
        
        # Handle "has" relation grammar fixes
        if r == "has":
            if "two children" in d.lower():
                return f"• {s.title()} have two children"
            elif "three children" in d.lower():
                return f"• {s.title()} have three children"
        
        # Handle age with special formatting
        if r == "age":
            if "years old" in d.lower():
                return f"• {s.title()} is {d}"
            elif d.isdigit():
                return f"• {s.title()} is {d} years old"
            else:
                return f"• {s.title()}'s age is {d}"
        
        # Handle time/date relations
        if r == "time" or r == "date":
            return f"• {s.title()} occurred in {d}"
        
        # Handle "and" relations (conjunctions)
        if r == "and":
            return f"• {s.title()} and {d} are connected"
        
        # Handle quality/attribute relations
        if r == "quality":
            return f"• {s.title()} is {d}"
        
        # Handle capital city relations
        if r == "capital":
            return f"• {d.title()} is the capital of {s.title()}"
        
        # Handle CEO/founder special relations
        if "ceo" in s_lower or "founder" in s_lower:
            if r == "modified_by" or r == "of":
                return f"• {s.title()} of {d}"
        
        return None
    
    def _conjugate_verb(self, verb: str, subject: str) -> str:
        """Conjugate verb based on subject."""
        # Special handling for "has/have"
        if verb in ["has", "have"]:
            if subject.lower() in ["you", "i", "we", "they"]:
                return "have"
            else:
                return "has"
        
        # Special handling for "is/are"
        if verb in ["is", "are", "am"]:
            if subject.lower() == "i":
                return "am"
            elif subject.lower() in ["you", "we", "they"]:
                return "are"
            else:
                return "is"
        
        # General verb conjugation
        if subject.lower() in ["you", "i", "we", "they"]:
            # Remove 's' for plural/second person
            if verb.endswith("s") and not verb.endswith("ss"):
                return verb[:-1]
        elif subject.lower() not in ["he", "she", "it"] and not verb.endswith("s"):
            # Add 's' for third person singular
            if verb.endswith("y") and verb[-2] not in "aeiou":
                return verb[:-1] + "ies"
            elif verb.endswith(("ch", "sh", "x", "z", "o")):
                return verb + "es"
            else:
                return verb + "s"
        return verb
    
    def _is_verb(self, word: str) -> bool:
        """Check if a word is likely a verb."""
        verb_endings = ["ed", "ing", "es", "s"]
        common_verbs = {
            "is", "was", "are", "were", "have", "has", "had",
            "do", "does", "did", "make", "makes", "made",
            "go", "goes", "went", "take", "takes", "took",
            "get", "gets", "got", "give", "gives", "gave",
            "find", "finds", "found", "tell", "tells", "told",
            "become", "becomes", "became", "seem", "seems", "seemed"
        }
        
        if word in common_verbs:
            return True
        
        # Check for verb-like endings
        for ending in verb_endings:
            if word.endswith(ending) and len(word) > len(ending) + 2:
                return True
        
        return False
    
    def _add_contextual_info(self, bullet: str, s: str, r: str, d: str) -> str:
        """Add contextual information to make bullets more informative."""
        # Add temporal context for past tense
        if any(past in r for past in ["founded", "created", "discovered", "invented"]):
            if d.isdigit() and len(d) == 4:  # Year
                if "in " + d not in bullet:
                    bullet = bullet.rstrip(".") + f" in {d}"
        
        # Add location context
        if r in ["founded", "established", "created"] and "in" not in bullet:
            # Could add location if available in context
            pass
        
        # Ensure relationships are clear
        if "and" in s:
            # Handle compound subjects
            parts = s.split(" and ")
            if len(parts) == 2 and r in ["founded", "created", "discovered"]:
                bullet = bullet.replace(s, f"{parts[0]} and {parts[1]} together")
        
        return bullet
    
    def _finalize_bullet(self, bullet: str) -> str:
        """Finalize bullet formatting."""
        # Ensure it starts with bullet point
        if not bullet.startswith("•"):
            bullet = f"• {bullet}"
        
        # Capitalize first letter after bullet
        if len(bullet) > 2:
            bullet = bullet[:2] + bullet[2].upper() + bullet[3:] if len(bullet) > 3 else bullet
        
        # Remove duplicate spaces
        bullet = " ".join(bullet.split())
        
        # Ensure no trailing punctuation unless it's meaningful
        bullet = bullet.rstrip(",;")
        
        return bullet


def enhance_memory_bullets(triples, formatter=None):
    """
    Enhance a list of triples into detailed bullet points.
    
    Args:
        triples: List of (subject, relation, object) tuples
        formatter: Optional EnhancedBulletFormatter instance
        
    Returns:
        List of formatted bullet strings
    """
    if formatter is None:
        formatter = EnhancedBulletFormatter()
    
    bullets = []
    seen = set()
    
    for triple in triples:
        if len(triple) >= 3:
            s, r, d = triple[:3]
            # Create unique key for deduplication
            key = (s.lower(), r.lower(), d.lower())
            if key not in seen:
                seen.add(key)
                bullet = formatter.format_bullet(s, r, d)
                bullets.append(bullet)
    
    return bullets


if __name__ == "__main__":
    # Test the enhanced formatter
    formatter = EnhancedBulletFormatter()
    
    test_triples = [
        ("Steve Jobs", "founded", "Apple"),
        ("Steve Wozniak", "founded", "Apple"),
        ("you", "has", "brother"),
        ("brother", "lives_in", "Portland"),
        ("brother", "teaches_at", "Reed College"),
        ("Marie Curie", "discovered", "radium"),
        ("Elon Musk", "works_at", "Tesla"),
        ("John", "studied", "computer science"),
        ("John", "graduated_from", "MIT"),
        ("sister", "age", "28"),
        ("Barcelona", "capital", "Catalonia"),
        ("professor", "published", "book about AI"),
    ]
    
    print("Enhanced Bullet Points:")
    print("=" * 50)
    
    for s, r, d in test_triples:
        bullet = formatter.format_bullet(s, r, d)
        print(bullet)
    
    print("\n" + "=" * 50)
    print("Batch Enhancement:")
    bullets = enhance_memory_bullets(test_triples)
    for bullet in bullets:
        print(bullet)