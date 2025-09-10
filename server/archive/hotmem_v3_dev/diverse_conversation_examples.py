"""
Diverse Conversation Examples for HotMem V4 Relation Extractor
Covers the wide range of topics people discuss in real conversations
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ConversationExample:
    """Single conversation example with entities and relations"""
    text: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    category: str
    difficulty: str = "medium"

class DiverseConversationExamples:
    """Collection of diverse conversation examples across multiple domains"""
    
    def __init__(self):
        self.examples = []
        self.categories = {
            "daily_life": "Daily Life (Family, Friends, Home)",
            "work_career": "Work & Career (Jobs, Business, Education)",
            "entertainment": "Entertainment (Movies, Music, Games)",
            "sports_hobbies": "Sports & Hobbies (Sports, Fitness, Hobbies)",
            "food_travel": "Food & Travel (Food, Travel)",
            "news_events": "News & Events (Politics, Current Events)"
        }
        self.relation_types = self._get_relation_types()
        
    def _get_relation_types(self) -> Dict[str, List[str]]:
        """Get relation types organized by category"""
        return {
            "daily_life": [
                "lives_in", "married_to", "dating", "engaged_to", "divorced_from",
                "parent_of", "child_of", "sibling_of", "grandparent_of", "grandchild_of",
                "takes_care_of", "cares_for", "lives_with", "roommate_of", "neighbor_of",
                "friends_with", "best_friends_with", "hangs_out_with", "introduced_to",
                "met_through", "knows", "famous_for", "known_for"
            ],
            "work_career": [
                "works_at", "employed_by", "works_for", "CEO_of", "founder_of", "owner_of",
                "manager_of", "supervises", "reports_to", "colleague_of", "teammate_of",
                "teaches_at", "studies_at", "majors_in", "graduated_from", "degree_in",
                "expert_in", "specializes_in", "consults_for", "advises", "mentors",
                "client_of", "customer_of", "serves", "represents", "speaks_at"
            ],
            "entertainment": [
                "directed", "produced", "wrote", "starred_in", "acted_in", "played_in",
                "sang_in", "composed", "performed_in", "hosted", "presented", "appeared_in",
                "won", "nominated_for", "awarded", "created", "designed", "developed",
                "published", "released", "streamed", "watched", "listened_to", "played",
                "reviewed", "rated", "criticized", "praised", "recommended"
            ],
            "sports_hobbies": [
                "plays_for", "coaches", "captain_of", "member_of", "fans_of", "supports",
                "cheer_for", "competes_in", "participates_in", "won", "lost", "defeated",
                "scored", "trained", "practices", "owns", "collects", "creates", "builds",
                "paints", "draws", "writes", "photographs", "gardens", "cooks", "bakes",
                "exercises", "runs", "swims", "cycles", "climbs", "hikes", "travels_to"
            ],
            "food_travel": [
                "cooks", "bakes", "prepares", "serves", "owns", "manages", "works_at",
                "visits", "travels_to", "flies_to", "drives_to", "stays_at", "books",
                "reserves", "reviews", "recommends", "criticizes", "popular_in", "famous_for",
                "located_in", "found_in", "grown_in", "produced_in", "imported_from",
                "exported_to", "sold_in", "available_in", "specializes_in", "known_for"
            ],
            "news_events": [
                "elected", "appointed", "resigned", "fired", "retired", "died", "born",
                "married", "divorced", "arrested", "convicted", "sentenced", "released",
                "discovered", "invented", "created", "launched", "announced", "reported",
                "stated", "claimed", "denied", "admitted", "confessed", "testified",
                "protested", "marched", "demonstrated", "struck", "boycotted", "supported",
                "opposed", "criticized", "praised", "awarded", "honored", "recognized"
            ]
        }
    
    def generate_daily_life_examples(self) -> List[ConversationExample]:
        """Generate daily life conversation examples"""
        examples = [
            ConversationExample(
                text="Sarah lives in New York with her husband Michael and their two kids.",
                entities=[
                    {"text": "Sarah", "type": "PERSON", "confidence": 0.95},
                    {"text": "New York", "type": "LOCATION", "confidence": 0.90},
                    {"text": "Michael", "type": "PERSON", "confidence": 0.95},
                    {"text": "kids", "type": "PERSON", "confidence": 0.85}
                ],
                relations=[
                    {"subject": "Sarah", "predicate": "lives_in", "object": "New York", "confidence": 0.90},
                    {"subject": "Sarah", "predicate": "married_to", "object": "Michael", "confidence": 0.95},
                    {"subject": "Sarah", "predicate": "parent_of", "object": "kids", "confidence": 0.85},
                    {"subject": "Michael", "predicate": "parent_of", "object": "kids", "confidence": 0.85},
                    {"subject": "Sarah", "predicate": "lives_with", "object": "Michael", "confidence": 0.90}
                ],
                category="daily_life"
            ),
            ConversationExample(
                text="My neighbor John takes care of his elderly mother who lives next door.",
                entities=[
                    {"text": "John", "type": "PERSON", "confidence": 0.95},
                    {"text": "mother", "type": "PERSON", "confidence": 0.85},
                    {"text": "next door", "type": "LOCATION", "confidence": 0.80}
                ],
                relations=[
                    {"subject": "John", "predicate": "neighbor_of", "object": "speaker", "confidence": 0.90},
                    {"subject": "John", "predicate": "takes_care_of", "object": "mother", "confidence": 0.95},
                    {"subject": "mother", "predicate": "lives_in", "object": "next door", "confidence": 0.80},
                    {"subject": "John", "predicate": "child_of", "object": "mother", "confidence": 0.85}
                ],
                category="daily_life"
            ),
            ConversationExample(
                text="Emma and her best friend Lisa met through college and now hang out every weekend.",
                entities=[
                    {"text": "Emma", "type": "PERSON", "confidence": 0.95},
                    {"text": "Lisa", "type": "PERSON", "confidence": 0.95},
                    {"text": "college", "type": "ORGANIZATION", "confidence": 0.85}
                ],
                relations=[
                    {"subject": "Emma", "predicate": "best_friends_with", "object": "Lisa", "confidence": 0.95},
                    {"subject": "Emma", "predicate": "met_through", "object": "college", "confidence": 0.85},
                    {"subject": "Lisa", "predicate": "met_through", "object": "college", "confidence": 0.85},
                    {"subject": "Emma", "predicate": "hangs_out_with", "object": "Lisa", "confidence": 0.90}
                ],
                category="daily_life"
            ),
            ConversationExample(
                text="David is famous for his amazing BBQ parties that he hosts in his backyard.",
                entities=[
                    {"text": "David", "type": "PERSON", "confidence": 0.95},
                    {"text": "BBQ parties", "type": "EVENT", "confidence": 0.85},
                    {"text": "backyard", "type": "LOCATION", "confidence": 0.80}
                ],
                relations=[
                    {"subject": "David", "predicate": "famous_for", "object": "BBQ parties", "confidence": 0.90},
                    {"subject": "David", "predicate": "hosts", "object": "BBQ parties", "confidence": 0.95},
                    {"subject": "BBQ parties", "predicate": "located_in", "object": "backyard", "confidence": 0.80}
                ],
                category="daily_life"
            ),
            ConversationExample(
                text="My roommate Alex studies at Stanford and is majoring in computer science.",
                entities=[
                    {"text": "Alex", "type": "PERSON", "confidence": 0.95},
                    {"text": "Stanford", "type": "ORGANIZATION", "confidence": 0.90},
                    {"text": "computer science", "type": "MISC", "confidence": 0.85}
                ],
                relations=[
                    {"subject": "Alex", "predicate": "roommate_of", "object": "speaker", "confidence": 0.90},
                    {"subject": "Alex", "predicate": "studies_at", "object": "Stanford", "confidence": 0.95},
                    {"subject": "Alex", "predicate": "majors_in", "object": "computer science", "confidence": 0.90}
                ],
                category="daily_life"
            )
        ]
        return examples
    
    def generate_work_career_examples(self) -> List[ConversationExample]:
        """Generate work and career conversation examples"""
        examples = [
            ConversationExample(
                text="Jennifer works at Google as a software engineer and reports to the engineering manager.",
                entities=[
                    {"text": "Jennifer", "type": "PERSON", "confidence": 0.95},
                    {"text": "Google", "type": "ORGANIZATION", "confidence": 0.90},
                    {"text": "software engineer", "type": "TITLE", "confidence": 0.85},
                    {"text": "engineering manager", "type": "TITLE", "confidence": 0.85}
                ],
                relations=[
                    {"subject": "Jennifer", "predicate": "works_at", "object": "Google", "confidence": 0.95},
                    {"subject": "Jennifer", "predicate": "employed_by", "object": "Google", "confidence": 0.90},
                    {"subject": "Jennifer", "predicate": "reports_to", "object": "engineering manager", "confidence": 0.85}
                ],
                category="work_career"
            ),
            ConversationExample(
                text="Professor Williams teaches at Harvard and specializes in artificial intelligence research.",
                entities=[
                    {"text": "Professor Williams", "type": "PERSON", "confidence": 0.95},
                    {"text": "Harvard", "type": "ORGANIZATION", "confidence": 0.90},
                    {"text": "artificial intelligence research", "type": "MISC", "confidence": 0.85}
                ],
                relations=[
                    {"subject": "Professor Williams", "predicate": "teaches_at", "object": "Harvard", "confidence": 0.95},
                    {"subject": "Professor Williams", "predicate": "specializes_in", "object": "artificial intelligence research", "confidence": 0.90}
                ],
                category="work_career"
            ),
            ConversationExample(
                text="Mark founded his own startup and now serves as CEO while mentoring young entrepreneurs.",
                entities=[
                    {"text": "Mark", "type": "PERSON", "confidence": 0.95},
                    {"text": "startup", "type": "ORGANIZATION", "confidence": 0.85},
                    {"text": "entrepreneurs", "type": "PERSON", "confidence": 0.80}
                ],
                relations=[
                    {"subject": "Mark", "predicate": "founder_of", "object": "startup", "confidence": 0.95},
                    {"subject": "Mark", "predicate": "CEO_of", "object": "startup", "confidence": 0.90},
                    {"subject": "Mark", "predicate": "mentors", "object": "entrepreneurs", "confidence": 0.85}
                ],
                category="work_career"
            ),
            ConversationExample(
                text="Dr. Smith graduated from MIT with a degree in physics and now consults for NASA.",
                entities=[
                    {"text": "Dr. Smith", "type": "PERSON", "confidence": 0.95},
                    {"text": "MIT", "type": "ORGANIZATION", "confidence": 0.90},
                    {"text": "physics", "type": "MISC", "confidence": 0.85},
                    {"text": "NASA", "type": "ORGANIZATION", "confidence": 0.90}
                ],
                relations=[
                    {"subject": "Dr. Smith", "predicate": "graduated_from", "object": "MIT", "confidence": 0.95},
                    {"subject": "Dr. Smith", "predicate": "degree_in", "object": "physics", "confidence": 0.85},
                    {"subject": "Dr. Smith", "predicate": "consults_for", "object": "NASA", "confidence": 0.90}
                ],
                category="work_career"
            ),
            ConversationExample(
                text="Sarah and Tom are colleagues at Microsoft who work on the same team.",
                entities=[
                    {"text": "Sarah", "type": "PERSON", "confidence": 0.95},
                    {"text": "Tom", "type": "PERSON", "confidence": 0.95},
                    {"text": "Microsoft", "type": "ORGANIZATION", "confidence": 0.90},
                    {"text": "team", "type": "ORGANIZATION", "confidence": 0.80}
                ],
                relations=[
                    {"subject": "Sarah", "predicate": "colleague_of", "object": "Tom", "confidence": 0.95},
                    {"subject": "Sarah", "predicate": "works_at", "object": "Microsoft", "confidence": 0.95},
                    {"subject": "Tom", "predicate": "works_at", "object": "Microsoft", "confidence": 0.95},
                    {"subject": "Sarah", "predicate": "teammate_of", "object": "Tom", "confidence": 0.90},
                    {"subject": "Sarah", "predicate": "member_of", "object": "team", "confidence": 0.85},
                    {"subject": "Tom", "predicate": "member_of", "object": "team", "confidence": 0.85}
                ],
                category="work_career"
            )
        ]
        return examples
    
    def generate_entertainment_examples(self) -> List[ConversationExample]:
        """Generate entertainment conversation examples"""
        examples = [
            ConversationExample(
                text="Christopher Nolan directed The Dark Knight trilogy which starred Christian Bale as Batman.",
                entities=[
                    {"text": "Christopher Nolan", "type": "PERSON", "confidence": 0.95},
                    {"text": "The Dark Knight trilogy", "type": "MISC", "confidence": 0.90},
                    {"text": "Christian Bale", "type": "PERSON", "confidence": 0.95},
                    {"text": "Batman", "type": "PERSON", "confidence": 0.90}
                ],
                relations=[
                    {"subject": "Christopher Nolan", "predicate": "directed", "object": "The Dark Knight trilogy", "confidence": 0.95},
                    {"subject": "Christian Bale", "predicate": "starred_in", "object": "The Dark Knight trilogy", "confidence": 0.95},
                    {"subject": "Christian Bale", "predicate": "played_in", "object": "The Dark Knight trilogy", "confidence": 0.90}
                ],
                category="entertainment"
            ),
            ConversationExample(
                text="Taylor Swift performed at the Grammy Awards and won Album of the Year.",
                entities=[
                    {"text": "Taylor Swift", "type": "PERSON", "confidence": 0.95},
                    {"text": "Grammy Awards", "type": "EVENT", "confidence": 0.90},
                    {"text": "Album of the Year", "type": "MISC", "confidence": 0.85}
                ],
                relations=[
                    {"subject": "Taylor Swift", "predicate": "performed_in", "object": "Grammy Awards", "confidence": 0.95},
                    {"subject": "Taylor Swift", "predicate": "won", "object": "Album of the Year", "confidence": 0.95}
                ],
                category="entertainment"
            ),
            ConversationExample(
                text="Morgan Freeman narrated the documentary March of the Penguins which was directed by Luc Jacquet.",
                entities=[
                    {"text": "Morgan Freeman", "type": "PERSON", "confidence": 0.95},
                    {"text": "March of the Penguins", "type": "MISC", "confidence": 0.90},
                    {"text": "Luc Jacquet", "type": "PERSON", "confidence": 0.95}
                ],
                relations=[
                    {"subject": "Morgan Freeman", "predicate": "narrated", "object": "March of the Penguins", "confidence": 0.95},
                    {"subject": "Luc Jacquet", "predicate": "directed", "object": "March of the Penguins", "confidence": 0.95}
                ],
                category="entertainment"
            ),
            ConversationExample(
                text="I watched The Crown on Netflix and listened to the soundtrack composed by Hans Zimmer.",
                entities=[
                    {"text": "The Crown", "type": "MISC", "confidence": 0.90},
                    {"text": "Netflix", "type": "ORGANIZATION", "confidence": 0.90},
                    {"text": "soundtrack", "type": "MISC", "confidence": 0.85},
                    {"text": "Hans Zimmer", "type": "PERSON", "confidence": 0.95}
                ],
                relations=[
                    {"subject": "speaker", "predicate": "watched", "object": "The Crown", "confidence": 0.90},
                    {"subject": "The Crown", "predicate": "streamed_on", "object": "Netflix", "confidence": 0.85},
                    {"subject": "speaker", "predicate": "listened_to", "object": "soundtrack", "confidence": 0.85},
                    {"subject": "Hans Zimmer", "predicate": "composed", "object": "soundtrack", "confidence": 0.95}
                ],
                category="entertainment"
            ),
            ConversationExample(
                text="Stephen King wrote The Shining which Stanley Kubrick later adapted into a film starring Jack Nicholson.",
                entities=[
                    {"text": "Stephen King", "type": "PERSON", "confidence": 0.95},
                    {"text": "The Shining", "type": "MISC", "confidence": 0.90},
                    {"text": "Stanley Kubrick", "type": "PERSON", "confidence": 0.95},
                    {"text": "film", "type": "MISC", "confidence": 0.80},
                    {"text": "Jack Nicholson", "type": "PERSON", "confidence": 0.95}
                ],
                relations=[
                    {"subject": "Stephen King", "predicate": "wrote", "object": "The Shining", "confidence": 0.95},
                    {"subject": "Stanley Kubrick", "predicate": "adapted", "object": "The Shining", "confidence": 0.90},
                    {"subject": "Stanley Kubrick", "predicate": "directed", "object": "film", "confidence": 0.85},
                    {"subject": "Jack Nicholson", "predicate": "starred_in", "object": "film", "confidence": 0.90}
                ],
                category="entertainment"
            )
        ]
        return examples
    
    def generate_sports_hobbies_examples(self) -> List[ConversationExample]:
        """Generate sports and hobbies conversation examples"""
        examples = [
            ConversationExample(
                text="LeBron James plays for the Lakers and has won four NBA championships.",
                entities=[
                    {"text": "LeBron James", "type": "PERSON", "confidence": 0.95},
                    {"text": "Lakers", "type": "ORGANIZATION", "confidence": 0.90},
                    {"text": "four NBA championships", "type": "MISC", "confidence": 0.85}
                ],
                relations=[
                    {"subject": "LeBron James", "predicate": "plays_for", "object": "Lakers", "confidence": 0.95},
                    {"subject": "LeBron James", "predicate": "won", "object": "four NBA championships", "confidence": 0.90}
                ],
                category="sports_hobbies"
            ),
            ConversationExample(
                text="My brother runs marathons and trains every morning at the local park.",
                entities=[
                    {"text": "brother", "type": "PERSON", "confidence": 0.85},
                    {"text": "marathons", "type": "EVENT", "confidence": 0.80},
                    {"text": "local park", "type": "LOCATION", "confidence": 0.80}
                ],
                relations=[
                    {"subject": "brother", "predicate": "runs", "object": "marathons", "confidence": 0.85},
                    {"subject": "brother", "predicate": "trains", "object": "local park", "confidence": 0.85}
                ],
                category="sports_hobbies"
            ),
            ConversationExample(
                text="Emma paints beautiful landscapes and exhibits her work at local art galleries.",
                entities=[
                    {"text": "Emma", "type": "PERSON", "confidence": 0.95},
                    {"text": "landscapes", "type": "MISC", "confidence": 0.85},
                    {"text": "local art galleries", "type": "LOCATION", "confidence": 0.80}
                ],
                relations=[
                    {"subject": "Emma", "predicate": "paints", "object": "landscapes", "confidence": 0.90},
                    {"subject": "Emma", "predicate": "exhibits", "object": "work", "confidence": 0.85},
                    {"subject": "work", "predicate": "located_in", "object": "local art galleries", "confidence": 0.80}
                ],
                category="sports_hobbies"
            ),
            ConversationExample(
                text="Tom Brady coached the Buccaneers after retiring from playing football.",
                entities=[
                    {"text": "Tom Brady", "type": "PERSON", "confidence": 0.95},
                    {"text": "Buccaneers", "type": "ORGANIZATION", "confidence": 0.90},
                    {"text": "football", "type": "MISC", "confidence": 0.85}
                ],
                relations=[
                    {"subject": "Tom Brady", "predicate": "coaches", "object": "Buccaneers", "confidence": 0.95},
                    {"subject": "Tom Brady", "predicate": "retired_from", "object": "football", "confidence": 0.90}
                ],
                category="sports_hobbies"
            ),
            ConversationExample(
                text="Sarah collects vintage cameras and photographs wildlife in her free time.",
                entities=[
                    {"text": "Sarah", "type": "PERSON", "confidence": 0.95},
                    {"text": "vintage cameras", "type": "MISC", "confidence": 0.85},
                    {"text": "wildlife", "type": "MISC", "confidence": 0.80}
                ],
                relations=[
                    {"subject": "Sarah", "predicate": "collects", "object": "vintage cameras", "confidence": 0.90},
                    {"subject": "Sarah", "predicate": "photographs", "object": "wildlife", "confidence": 0.85}
                ],
                category="sports_hobbies"
            )
        ]
        return examples
    
    def generate_food_travel_examples(self) -> List[ConversationExample]:
        """Generate food and travel conversation examples"""
        examples = [
            ConversationExample(
                text="Gordon Ramsay owns several Michelin-starred restaurants and is famous for his cooking shows.",
                entities=[
                    {"text": "Gordon Ramsay", "type": "PERSON", "confidence": 0.95},
                    {"text": "Michelin-starred restaurants", "type": "ORGANIZATION", "confidence": 0.85},
                    {"text": "cooking shows", "type": "MISC", "confidence": 0.80}
                ],
                relations=[
                    {"subject": "Gordon Ramsay", "predicate": "owns", "object": "Michelin-starred restaurants", "confidence": 0.95},
                    {"subject": "Gordon Ramsay", "predicate": "famous_for", "object": "cooking shows", "confidence": 0.90}
                ],
                category="food_travel"
            ),
            ConversationExample(
                text="Last summer, I traveled to Japan and stayed at a traditional ryokan in Kyoto.",
                entities=[
                    {"text": "summer", "type": "DATE", "confidence": 0.80},
                    {"text": "Japan", "type": "LOCATION", "confidence": 0.90},
                    {"text": "traditional ryokan", "type": "LOCATION", "confidence": 0.85},
                    {"text": "Kyoto", "type": "LOCATION", "confidence": 0.90}
                ],
                relations=[
                    {"subject": "speaker", "predicate": "traveled_to", "object": "Japan", "confidence": 0.95},
                    {"subject": "speaker", "predicate": "stayed_at", "object": "traditional ryokan", "confidence": 0.90},
                    {"subject": "traditional ryokan", "predicate": "located_in", "object": "Kyoto", "confidence": 0.85}
                ],
                category="food_travel"
            ),
            ConversationExample(
                text="The local Italian restaurant is known for their homemade pasta and wood-fired pizza.",
                entities=[
                    {"text": "local Italian restaurant", "type": "ORGANIZATION", "confidence": 0.85},
                    {"text": "homemade pasta", "type": "MISC", "confidence": 0.80},
                    {"text": "wood-fired pizza", "type": "MISC", "confidence": 0.80}
                ],
                relations=[
                    {"subject": "local Italian restaurant", "predicate": "known_for", "object": "homemade pasta", "confidence": 0.85},
                    {"subject": "local Italian restaurant", "predicate": "known_for", "object": "wood-fired pizza", "confidence": 0.85}
                ],
                category="food_travel"
            ),
            ConversationExample(
                text="Maria bakes amazing sourdough bread that she sells at the farmers market.",
                entities=[
                    {"text": "Maria", "type": "PERSON", "confidence": 0.95},
                    {"text": "sourdough bread", "type": "MISC", "confidence": 0.85},
                    {"text": "farmers market", "type": "LOCATION", "confidence": 0.80}
                ],
                relations=[
                    {"subject": "Maria", "predicate": "bakes", "object": "sourdough bread", "confidence": 0.95},
                    {"subject": "Maria", "predicate": "sells", "object": "sourdough bread", "confidence": 0.90},
                    {"subject": "sourdough bread", "predicate": "sold_in", "object": "farmers market", "confidence": 0.85}
                ],
                category="food_travel"
            ),
            ConversationExample(
                text="We flew to Paris and visited the Eiffel Tower before booking a table at a Michelin-starred restaurant.",
                entities=[
                    {"text": "Paris", "type": "LOCATION", "confidence": 0.90},
                    {"text": "Eiffel Tower", "type": "LOCATION", "confidence": 0.90},
                    {"text": "Michelin-starred restaurant", "type": "ORGANIZATION", "confidence": 0.85}
                ],
                relations=[
                    {"subject": "speaker", "predicate": "flew_to", "object": "Paris", "confidence": 0.95},
                    {"subject": "speaker", "predicate": "visited", "object": "Eiffel Tower", "confidence": 0.95},
                    {"subject": "speaker", "predicate": "booked", "object": "table", "confidence": 0.85},
                    {"subject": "table", "predicate": "located_in", "object": "Michelin-starred restaurant", "confidence": 0.80}
                ],
                category="food_travel"
            )
        ]
        return examples
    
    def generate_news_events_examples(self) -> List[ConversationExample]:
        """Generate news and events conversation examples"""
        examples = [
            ConversationExample(
                text="President Biden was elected in 2020 and appointed several cabinet members.",
                entities=[
                    {"text": "President Biden", "type": "PERSON", "confidence": 0.95},
                    {"text": "2020", "type": "DATE", "confidence": 0.90},
                    {"text": "cabinet members", "type": "PERSON", "confidence": 0.80}
                ],
                relations=[
                    {"subject": "President Biden", "predicate": "elected", "object": "2020", "confidence": 0.95},
                    {"subject": "President Biden", "predicate": "appointed", "object": "cabinet members", "confidence": 0.90}
                ],
                category="news_events"
            ),
            ConversationExample(
                text="Scientists discovered a new species of butterfly in the Amazon rainforest.",
                entities=[
                    {"text": "Scientists", "type": "PERSON", "confidence": 0.85},
                    {"text": "new species of butterfly", "type": "MISC", "confidence": 0.85},
                    {"text": "Amazon rainforest", "type": "LOCATION", "confidence": 0.90}
                ],
                relations=[
                    {"subject": "Scientists", "predicate": "discovered", "object": "new species of butterfly", "confidence": 0.95},
                    {"subject": "new species of butterfly", "predicate": "found_in", "object": "Amazon rainforest", "confidence": 0.90}
                ],
                category="news_events"
            ),
            ConversationExample(
                text="Thousands of people marched in the climate change protest organized by environmental activists.",
                entities=[
                    {"text": "Thousands of people", "type": "PERSON", "confidence": 0.85},
                    {"text": "climate change protest", "type": "EVENT", "confidence": 0.85},
                    {"text": "environmental activists", "type": "PERSON", "confidence": 0.80}
                ],
                relations=[
                    {"subject": "Thousands of people", "predicate": "marched", "object": "climate change protest", "confidence": 0.95},
                    {"subject": "environmental activists", "predicate": "organized", "object": "climate change protest", "confidence": 0.90}
                ],
                category="news_events"
            ),
            ConversationExample(
                text="Apple announced their new iPhone at the annual product launch event in Cupertino.",
                entities=[
                    {"text": "Apple", "type": "ORGANIZATION", "confidence": 0.90},
                    {"text": "new iPhone", "type": "PRODUCT", "confidence": 0.85},
                    {"text": "annual product launch event", "type": "EVENT", "confidence": 0.80},
                    {"text": "Cupertino", "type": "LOCATION", "confidence": 0.85}
                ],
                relations=[
                    {"subject": "Apple", "predicate": "announced", "object": "new iPhone", "confidence": 0.95},
                    {"subject": "Apple", "predicate": "launched", "object": "new iPhone", "confidence": 0.90},
                    {"subject": "annual product launch event", "predicate": "located_in", "object": "Cupertino", "confidence": 0.85}
                ],
                category="news_events"
            ),
            ConversationExample(
                text="The Nobel Prize in Physics was awarded to scientists who discovered black hole formations.",
                entities=[
                    {"text": "Nobel Prize in Physics", "type": "MISC", "confidence": 0.90},
                    {"text": "scientists", "type": "PERSON", "confidence": 0.85},
                    {"text": "black hole formations", "type": "MISC", "confidence": 0.85}
                ],
                relations=[
                    {"subject": "Nobel Prize in Physics", "predicate": "awarded_to", "object": "scientists", "confidence": 0.95},
                    {"subject": "scientists", "predicate": "discovered", "object": "black hole formations", "confidence": 0.90}
                ],
                category="news_events"
            )
        ]
        return examples
    
    def generate_all_examples(self) -> List[ConversationExample]:
        """Generate all diverse conversation examples"""
        all_examples = []
        
        # Generate examples from each category
        all_examples.extend(self.generate_daily_life_examples())
        all_examples.extend(self.generate_work_career_examples())
        all_examples.extend(self.generate_entertainment_examples())
        all_examples.extend(self.generate_sports_hobbies_examples())
        all_examples.extend(self.generate_food_travel_examples())
        all_examples.extend(self.generate_news_events_examples())
        
        self.examples = all_examples
        return all_examples
    
    def export_to_json(self, filename: str = "diverse_conversation_examples.json") -> None:
        """Export examples to JSON file"""
        examples_data = []
        for example in self.examples:
            examples_data.append({
                "text": example.text,
                "entities": example.entities,
                "relations": example.relations,
                "category": example.category,
                "difficulty": example.difficulty
            })
        
        with open(filename, 'w') as f:
            json.dump(examples_data, f, indent=2)
        
        print(f"Exported {len(examples_data)} examples to {filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the examples"""
        if not self.examples:
            self.generate_all_examples()
        
        stats = {
            "total_examples": len(self.examples),
            "categories": {},
            "relation_types": set(),
            "entity_types": set()
        }
        
        for example in self.examples:
            # Count by category
            if example.category not in stats["categories"]:
                stats["categories"][example.category] = 0
            stats["categories"][example.category] += 1
            
            # Collect relation types
            for relation in example.relations:
                stats["relation_types"].add(relation["predicate"])
            
            # Collect entity types
            for entity in example.entities:
                stats["entity_types"].add(entity["type"])
        
        stats["relation_types"] = sorted(list(stats["relation_types"]))
        stats["entity_types"] = sorted(list(stats["entity_types"]))
        
        return stats

def main():
    """Generate and display diverse conversation examples"""
    generator = DiverseConversationExamples()
    examples = generator.generate_all_examples()
    
    print("🎯 DIVERSE CONVERSATION EXAMPLES FOR HOTMEM V4")
    print("=" * 60)
    print()
    
    # Display statistics
    stats = generator.get_statistics()
    print(f"📊 STATISTICS:")
    print(f"   Total Examples: {stats['total_examples']}")
    print(f"   Categories: {len(stats['categories'])}")
    print(f"   Relation Types: {len(stats['relation_types'])}")
    print(f"   Entity Types: {len(stats['entity_types'])}")
    print()
    
    # Display examples by category
    for category_name, category_display in generator.categories.items():
        category_examples = [ex for ex in examples if ex.category == category_name]
        print(f"📋 {category_display} ({len(category_examples)} examples):")
        print()
        
        for i, example in enumerate(category_examples[:3], 1):  # Show first 3 examples
            print(f"   {i}. {example.text}")
            if example.relations:
                relations_str = ", ".join([f"{r['subject']} {r['predicate']} {r['object']}" for r in example.relations])
                print(f"      Relations: {relations_str}")
            print()
        
        if len(category_examples) > 3:
            print(f"   ... and {len(category_examples) - 3} more examples")
        print()
    
    # Export to JSON
    generator.export_to_json()
    
    print("🎯 RELATION TYPES BY CATEGORY:")
    for category, relations in generator.relation_types.items():
        print(f"   {generator.categories[category]}:")
        for relation in relations:
            print(f"     - {relation}")
        print()

if __name__ == "__main__":
    main()