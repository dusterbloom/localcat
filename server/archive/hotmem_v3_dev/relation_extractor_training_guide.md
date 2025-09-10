# Training Guide: HotMem V4 Relation Extractor LLM

## 🎯 Training Philosophy

**Conversational excellence over business focus.** We're training a relation extractor that understands the wide range of topics people discuss in real conversations, not just business relationships.

## 📊 Model Requirements

### Base Model
- **Model**: Qwen2.5-0.5B (already proven effective)
- **Why**: Perfect balance of capability and speed for real-time conversations
- **Context**: 2048 tokens (sufficient for relation extraction)

### Training Data Strategy
- **Quantity**: 10K high-quality examples (not 1M)
- **Focus**: 70+ diverse relation types across 6 conversation categories
- **Quality**: Expert-curated, conversational examples from real-world domains
- **Domains**: Daily Life, Work & Career, Entertainment, Sports & Hobbies, Food & Travel, News & Events

## 🎯 Key Relation Types to Master

### Daily Life Relations
- `lives_in`: Person → Location
- `married_to`: Person → Person
- `parent_of`: Person → Person
- `child_of`: Person → Person
- `takes_care_of`: Person → Person
- `friends_with`: Person → Person
- `hangs_out_with`: Person → Person
- `met_through`: Person → Entity
- `neighbor_of`: Person → Person
- `roommate_of`: Person → Person

### Work & Career Relations
- `works_at`: Person → Organization
- `employed_by`: Person → Organization
- `CEO_of`: Person → Organization
- `founder_of`: Person → Organization
- `teaches_at`: Person → Organization
- `studies_at`: Person → Organization
- `reports_to`: Person → Person
- `colleague_of`: Person → Person
- `mentors`: Person → Person
- `graduated_from`: Person → Organization

### Entertainment Relations
- `directed`: Person → Creative Work
- `starred_in`: Person → Creative Work
- `composed`: Person → Creative Work
- `performed_in`: Person → Event
- `won`: Person → Award
- `nominated_for`: Person → Award
- `watched`: Person → Creative Work
- `listened_to`: Person → Creative Work
- `played`: Person → Game
- `reviewed`: Person → Creative Work

### Sports & Hobbies Relations
- `plays_for`: Person → Team
- `coaches`: Person → Team
- `competes_in`: Person → Event
- `won`: Person → Competition
- `trains`: Person → Location/Activity
- `paints`: Person → Art
- `collects`: Person → Items
- `photographs`: Person → Subject
- `exercises`: Person → Activity
- `builds`: Person → Object

### Food & Travel Relations
- `cooks`: Person → Food
- `owns`: Person → Restaurant
- `traveled_to`: Person → Location
- `stayed_at`: Person → Accommodation
- `visited`: Person → Location
- `booked`: Person → Reservation
- `famous_for`: Entity → Attribute
- `known_for`: Entity → Attribute
- `located_in`: Entity → Location
- `popular_in`: Entity → Location

### News & Events Relations
- `elected`: Person → Position
- `appointed`: Person → Position
- `discovered`: Person → Discovery
- `announced`: Organization → Product/News
- `protested`: Person → Cause
- `marched`: Person → Event
- `awarded`: Organization → Person
- `arrested`: Person → Crime
- `died`: Person → Cause
- `born`: Person → Location/Date

## 📝 Training Data Format

### JSON Structure
```json
{
  "text": "Tim Cook is the CEO of Apple and is based in Cupertino.",
  "entities": [
    {"text": "Tim Cook", "type": "PERSON", "start": 0, "end": 9},
    {"text": "Apple", "type": "ORGANIZATION", "start": 21, "end": 26},
    {"text": "Cupertino", "type": "LOCATION", "start": 44, "end": 53}
  ],
  "relations": [
    {
      "subject": "Tim Cook",
      "predicate": "CEO_of",
      "object": "Apple",
      "confidence": 0.95
    },
    {
      "subject": "Apple", 
      "predicate": "headquartered_in",
      "object": "Cupertino",
      "confidence": 0.90
    }
  ]
}
```

### Example Sets by Conversation Category

#### Daily Life Examples
```json
{
  "category": "Daily Life",
  "examples": [
    {
      "text": "Sarah lives in New York with her husband Michael and their two kids.",
      "entities": [
        {"text": "Sarah", "type": "PERSON", "confidence": 0.95},
        {"text": "New York", "type": "LOCATION", "confidence": 0.90},
        {"text": "Michael", "type": "PERSON", "confidence": 0.95},
        {"text": "kids", "type": "PERSON", "confidence": 0.85}
      ],
      "relations": [
        {"subject": "Sarah", "predicate": "lives_in", "object": "New York", "confidence": 0.90},
        {"subject": "Sarah", "predicate": "married_to", "object": "Michael", "confidence": 0.95},
        {"subject": "Sarah", "predicate": "parent_of", "object": "kids", "confidence": 0.85},
        {"subject": "Michael", "predicate": "parent_of", "object": "kids", "confidence": 0.85}
      ]
    },
    {
      "text": "Emma and her best friend Lisa met through college and now hang out every weekend.",
      "entities": [
        {"text": "Emma", "type": "PERSON", "confidence": 0.95},
        {"text": "Lisa", "type": "PERSON", "confidence": 0.95},
        {"text": "college", "type": "ORGANIZATION", "confidence": 0.85}
      ],
      "relations": [
        {"subject": "Emma", "predicate": "best_friends_with", "object": "Lisa", "confidence": 0.95},
        {"subject": "Emma", "predicate": "met_through", "object": "college", "confidence": 0.85},
        {"subject": "Lisa", "predicate": "met_through", "object": "college", "confidence": 0.85},
        {"subject": "Emma", "predicate": "hangs_out_with", "object": "Lisa", "confidence": 0.90}
      ]
    }
  ]
}
```

#### Entertainment Examples
```json
{
  "category": "Entertainment",
  "examples": [
    {
      "text": "Christopher Nolan directed The Dark Knight trilogy which starred Christian Bale as Batman.",
      "entities": [
        {"text": "Christopher Nolan", "type": "PERSON", "confidence": 0.95},
        {"text": "The Dark Knight trilogy", "type": "MISC", "confidence": 0.90},
        {"text": "Christian Bale", "type": "PERSON", "confidence": 0.95},
        {"text": "Batman", "type": "PERSON", "confidence": 0.90}
      ],
      "relations": [
        {"subject": "Christopher Nolan", "predicate": "directed", "object": "The Dark Knight trilogy", "confidence": 0.95},
        {"subject": "Christian Bale", "predicate": "starred_in", "object": "The Dark Knight trilogy", "confidence": 0.95},
        {"subject": "Christian Bale", "predicate": "played_in", "object": "The Dark Knight trilogy", "confidence": 0.90}
      ]
    },
    {
      "text": "Taylor Swift performed at the Grammy Awards and won Album of the Year.",
      "entities": [
        {"text": "Taylor Swift", "type": "PERSON", "confidence": 0.95},
        {"text": "Grammy Awards", "type": "EVENT", "confidence": 0.90},
        {"text": "Album of the Year", "type": "MISC", "confidence": 0.85}
      ],
      "relations": [
        {"subject": "Taylor Swift", "predicate": "performed_in", "object": "Grammy Awards", "confidence": 0.95},
        {"subject": "Taylor Swift", "predicate": "won", "object": "Album of the Year", "confidence": 0.95}
      ]
    }
  ]
}
```

#### Food & Travel Examples
```json
{
  "category": "Food & Travel",
  "examples": [
    {
      "text": "Last summer, I traveled to Japan and stayed at a traditional ryokan in Kyoto.",
      "entities": [
        {"text": "summer", "type": "DATE", "confidence": 0.80},
        {"text": "Japan", "type": "LOCATION", "confidence": 0.90},
        {"text": "traditional ryokan", "type": "LOCATION", "confidence": 0.85},
        {"text": "Kyoto", "type": "LOCATION", "confidence": 0.90}
      ],
      "relations": [
        {"subject": "speaker", "predicate": "traveled_to", "object": "Japan", "confidence": 0.95},
        {"subject": "speaker", "predicate": "stayed_at", "object": "traditional ryokan", "confidence": 0.90},
        {"subject": "traditional ryokan", "predicate": "located_in", "object": "Kyoto", "confidence": 0.85}
      ]
    },
    {
      "text": "Maria bakes amazing sourdough bread that she sells at the farmers market.",
      "entities": [
        {"text": "Maria", "type": "PERSON", "confidence": 0.95},
        {"text": "sourdough bread", "type": "MISC", "confidence": 0.85},
        {"text": "farmers market", "type": "LOCATION", "confidence": 0.80}
      ],
      "relations": [
        {"subject": "Maria", "predicate": "bakes", "object": "sourdough bread", "confidence": 0.95},
        {"subject": "Maria", "predicate": "sells", "object": "sourdough bread", "confidence": 0.90},
        {"subject": "sourdough bread", "predicate": "sold_in", "object": "farmers market", "confidence": 0.85}
      ]
    }
  ]
}
```

## 🛠️ Training Process

### Phase 1: Data Preparation (Week 1)

#### Step 1: Download Base Datasets
```bash
# Download REBEL dataset (high-quality relation extraction)
git lfs install
git clone https://huggingface.co/datasets/Babelscape/rebel-dataset

# Download TACRED dataset (standard benchmark)
wget https://tacred-data.s3.amazonaws.com/tacred-data.tar.gz
tar -xzf tacred-data.tar.gz
```

#### Step 2: Process and Clean Data
```python
import json
from typing import List, Dict, Any

def process_rebel_data(rebel_file: str) -> List[Dict[str, Any]]:
    """Process REBEL dataset to our format"""
    processed_examples = []
    
    with open(rebel_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Convert REBEL format to our format
            example = {
                "text": data["text"],
                "entities": [],
                "relations": []
            }
            
            # Process entities
            for entity in data.get("entities", []):
                example["entities"].append({
                    "text": entity["text"],
                    "type": entity["type"],
                    "start": entity["start"],
                    "end": entity["end"]
                })
            
            # Process relations
            for relation in data.get("relations", []):
                example["relations"].append({
                    "subject": relation["subject"],
                    "predicate": relation["predicate"],
                    "object": relation["object"],
                    "confidence": 0.95
                })
            
            processed_examples.append(example)
    
    return processed_examples
```

#### Step 3: Generate Synthetic Examples
```python
import openai

def generate_conversational_examples(category: str, relation_types: List[str], count: int = 100) -> List[Dict[str, Any]]:
    """Generate diverse conversational examples using GPT-4"""
    
    category_descriptions = {
        "daily_life": "family, friends, home life, personal relationships",
        "work_career": "jobs, business, education, professional life",
        "entertainment": "movies, music, games, books, creative works",
        "sports_hobbies": "sports, fitness, hobbies, recreational activities",
        "food_travel": "cooking, restaurants, travel, cultural experiences",
        "news_events": "current events, politics, discoveries, awards"
    }
    
    prompt = f"""
    Generate {count} high-quality conversational examples for the {category} category.
    
    Category: {category_descriptions[category]}
    Target Relations: {', '.join(relation_types)}
    
    Requirements:
    1. Natural, conversational language people actually use
    2. Clear entity boundaries and types
    3. Unambiguous relations from the target list
    4. Diverse scenarios within this category
    5. First and third-person perspectives
    6. Present and past tenses
    7. Various sentence structures (simple, compound, complex)
    
    Format each example as:
    Text: [natural conversational sentence]
    Entities: [entity1] (TYPE), [entity2] (TYPE), ...
    Relations: [subject] [relation] [object], [subject2] [relation2] [object2], ...
    
    Focus on realistic conversations people have about {category_descriptions[category]}.
    Include personal stories, observations, and casual discussions.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse and format the response
    examples = parse_conversational_examples(response.choices[0].message.content, category)
    return examples

def generate_diverse_synthetic_data() -> List[Dict[str, Any]]:
    """Generate synthetic data across all conversation categories"""
    
    all_examples = []
    
    # Define categories and their relation types
    categories = {
        "daily_life": ["lives_in", "married_to", "parent_of", "friends_with", "hangs_out_with"],
        "work_career": ["works_at", "teaches_at", "CEO_of", "colleague_of", "mentors"],
        "entertainment": ["directed", "starred_in", "won", "watched", "listened_to"],
        "sports_hobbies": ["plays_for", "coaches", "competes_in", "paints", "collects"],
        "food_travel": ["cooks", "traveled_to", "stayed_at", "famous_for", "known_for"],
        "news_events": ["elected", "discovered", "announced", "protested", "awarded"]
    }
    
    # Generate examples for each category
    for category, relations in categories.items():
        category_examples = generate_conversational_examples(category, relations, 150)
        all_examples.extend(category_examples)
    
    return all_examples
```

#### Step 4: Create Diverse Training Set
```python
def create_diverse_training_set() -> List[Dict[str, Any]]:
    """Create 10K high-quality diverse conversational examples"""
    
    training_examples = []
    
    # Phase 1: Base dataset processing (4K)
    # Process REBEL and TACRED but filter for conversational relevance
    rebel_examples = process_rebel_data("rebel-dataset/train.jsonl")
    conversational_rebel = filter_conversational_examples(rebel_examples, 2500)
    
    tacred_examples = process_tacred_data("tacred/data/json/train.json")
    conversational_tacred = filter_conversational_examples(tacred_examples, 1500)
    
    training_examples.extend(conversational_rebel)
    training_examples.extend(conversational_tacred)
    
    # Phase 2: Diverse synthetic examples (4K)
    synthetic_examples = generate_diverse_synthetic_data()
    training_examples.extend(synthetic_examples)
    
    # Phase 3: Curated conversational examples (2K)
    # Load our diverse conversation examples
    from diverse_conversation_examples import DiverseConversationExamples
    
    generator = DiverseConversationExamples()
    curated_examples = generator.generate_all_examples()
    
    # Extend curated examples with variations
    extended_curated = create_curated_variations(curated_examples, 1500)
    training_examples.extend(extended_curated)
    
    # Phase 4: Real-world conversation data (1K)
    # Add transcripts from conversations, interviews, podcasts
    real_world_examples = process_real_world_conversations("conversations/", 1000)
    training_examples.extend(real_world_examples)
    
    # Balance across categories
    balanced_examples = balance_across_categories(training_examples)
    
    # Shuffle and validate
    random.shuffle(balanced_examples)
    validated_examples = validate_examples(balanced_examples)
    
    return validated_examples[:10000]  # Ensure exactly 10K examples

def filter_conversational_examples(examples: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
    """Filter examples for conversational relevance"""
    
    conversational_keywords = [
        "family", "friend", "home", "live", "work", "school", "movie", "music", 
        "game", "sport", "food", "travel", "news", "event", "personal", "I", "we", 
        "my", "our", "yesterday", "today", "weekend"
    ]
    
    filtered = []
    for example in examples:
        text = example["text"].lower()
        # Check if example contains conversational elements
        if any(keyword in text for keyword in conversational_keywords):
            # Check if it's not overly technical/business focused
            if not any(term in text for term in ["corporate", "revenue", "merger", "acquisition"]):
                filtered.append(example)
        if len(filtered) >= target_count:
            break
    
    return filtered

def create_curated_variations(base_examples: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
    """Create variations of curated examples to expand dataset"""
    
    variations = []
    variation_patterns = [
        lambda text: text.replace("lives in", "is living in"),
        lambda text: text.replace("works at", "is working at"),
        lambda text: text.replace("yesterday", "last week"),
        lambda text: text.replace("last summer", "this year"),
        lambda text: text.replace("my friend", "my best friend"),
    ]
    
    for example in base_examples:
        # Add original
        variations.append(example)
        
        # Add variations
        for pattern in variation_patterns:
            if len(variations) >= target_count:
                break
            
            try:
                varied_text = pattern(example["text"])
                varied_example = example.copy()
                varied_example["text"] = varied_text
                variations.append(varied_example)
            except:
                continue
        
        if len(variations) >= target_count:
            break
    
    return variations[:target_count]
```

### Phase 2: Model Training (Week 1-2)

#### Step 1: Configure Training Parameters
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configure tokenizer for relation extraction
tokenizer.add_special_tokens({
    "additional_special_tokens": ["[ENT]", "[REL]", "[/ENT]", "[/REL]"]
})
model.resize_token_embeddings(len(tokenizer))
```

#### Step 2: Create Conversational Training Prompts
```python
def create_conversational_training_prompt(example: Dict[str, Any], category: str) -> str:
    """Create structured training prompt for conversational understanding"""
    
    # Format entities
    entity_text = ""
    for entity in example["entities"]:
        entity_text += f"{entity['text']} ({entity['type']}), "
    
    # Format relations
    relation_text = ""
    for relation in example["relations"]:
        relation_text += f"{relation['subject']} --{relation['predicate']}--> {relation['object']}, "
    
    category_instructions = {
        "daily_life": "Focus on personal relationships, family connections, and daily activities.",
        "work_career": "Focus on professional relationships, employment, and educational connections.",
        "entertainment": "Focus on creative works, performances, awards, and media consumption.",
        "sports_hobbies": "Focus on sports participation, hobbies, collections, and recreational activities.",
        "food_travel": "Focus on culinary experiences, travel, accommodations, and cultural activities.",
        "news_events": "Focus on current events, discoveries, announcements, and significant life events."
    }
    
    prompt = f"""Extract entities and relations from this conversational text:

Text: {example['text']}

Entities: {entity_text.rstrip(', ')}

Relations: {relation_text.rstrip(', ')}

Context: This is a {category} conversation. {category_instructions.get(category, '')}

Task: Learn to identify relationships in natural conversations about everyday topics.
Focus on understanding how people describe relationships in casual speech.
Pay attention to conversational language, personal pronouns, and context.

Important: This text represents how people actually speak about relationships in real life.
"""
    
    return prompt

def create_category_specific_prompts() -> Dict[str, str]:
    """Create specialized prompts for each conversation category"""
    
    prompts = {
        "daily_life": """You are analyzing conversations about daily life, family, and friends.

Focus on these relation types:
- lives_in: Where someone resides
- married_to: Spousal relationships
- parent_of/child_of: Family relationships
- friends_with/best_friends_with: Friendships
- takes_care_of: Caregiving relationships
- hangs_out_with: Social relationships
- met_through: How people met
- neighbor_of/roommate_of: Living proximity relationships

Examples:
- "Sarah lives in New York with her husband Michael"
- "Emma and Lisa met through college and now hang out every weekend"
- "My neighbor John takes care of his elderly mother"

Pay special attention to family terms, locations, and social activities.""",
        
        "entertainment": """You are analyzing conversations about entertainment, movies, music, and games.

Focus on these relation types:
- directed/starred_in/acted_in: Film/TV relationships
- composed/sang/performed_in: Music relationships
- won/nominated_for: Award relationships
- watched/listened_to/played: Consumption relationships
- wrote/produced/created: Creative relationships

Examples:
- "Christopher Nolan directed The Dark Knight trilogy"
- "Taylor Swift performed at the Grammy Awards and won Album of the Year"
- "I watched the new Marvel movie last weekend"

Pay attention to creative works, performances, awards, and entertainment consumption.""",
        
        "food_travel": """You are analyzing conversations about food, travel, and cultural experiences.

Focus on these relation types:
- cooks/bakes/prepares: Food preparation
- traveled_to/stayed_at/visited: Travel relationships
- famous_for/known_for: Reputation relationships
- located_in/popular_in: Geographic relationships
- owns/manages: Business relationships

Examples:
- "Last summer I traveled to Japan and stayed at a traditional ryokan"
- "Maria bakes amazing sourdough bread that she sells at the farmers market"
- "The local Italian restaurant is known for their homemade pasta"

Pay attention to travel experiences, culinary activities, and cultural mentions."""
    }
    
    return prompts
```

#### Step 3: Configure Training
```python
training_args = TrainingArguments(
    output_dir="./relation-extractor-v3",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    learning_rate=2e-5,
    fp16=True,
    gradient_accumulation_steps=2,
)
```

#### Step 4: Create Dataset Class
```python
from torch.utils.data import Dataset

class RelationExtractionDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length=1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = create_training_prompt(example)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create target (same as input for causal LM)
        targets = inputs["input_ids"].clone()
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets.squeeze()
        }
```

#### Step 5: Train Model
```python
# Create dataset
train_dataset = RelationExtractionDataset(train_examples, tokenizer)
eval_dataset = RelationExtractionDataset(val_examples, tokenizer)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save model
trainer.save_model("./relation-extractor-v3-final")
tokenizer.save_pretrained("./relation-extractor-v3-final")
```

### Phase 3: Evaluation and Fine-tuning (Week 2)

#### Step 1: Create Conversational Evaluation Set
```python
def create_conversational_evaluation_set() -> List[Dict[str, Any]]:
    """Create diverse conversational evaluation set"""
    
    eval_examples = []
    
    # Include examples from all conversation categories
    categories = ["daily_life", "work_career", "entertainment", "sports_hobbies", "food_travel", "news_events"]
    
    for category in categories:
        # Create examples with varying complexity
        category_examples = []
        
        # Simple examples (clear relations)
        simple_examples = create_simple_category_examples(category, 20)
        category_examples.extend(simple_examples)
        
        # Complex examples (multiple relations)
        complex_examples = create_complex_category_examples(category, 20)
        category_examples.extend(complex_examples)
        
        # Conversational examples (natural speech patterns)
        conv_examples = create_conversational_category_examples(category, 20)
        category_examples.extend(conv_examples)
        
        # Ambiguous examples (require context)
        ambiguous_examples = create_ambiguous_category_examples(category, 10)
        category_examples.extend(ambiguous_examples)
        
        eval_examples.extend(category_examples)
    
    return eval_examples

def create_simple_category_examples(category: str, count: int) -> List[Dict[str, Any]]:
    """Create simple examples with clear relations"""
    
    # Use predefined templates for each category
    templates = {
        "daily_life": [
            "{person} lives in {location}",
            "{person} is married to {person}",
            "{person} and {person} are friends"
        ],
        "entertainment": [
            "{person} directed {movie}",
            "{person} won {award}",
            "{person} starred in {movie}"
        ],
        "food_travel": [
            "{person} traveled to {location}",
            "{person} cooks {food}",
            "{restaurant} is famous for {dish}"
        ]
    }
    
    examples = []
    category_templates = templates.get(category, templates["daily_life"])
    
    for i in range(count):
        template = random.choice(category_templates)
        # Fill template with appropriate entities
        example = fill_template(template, category)
        examples.append(example)
    
    return examples

def create_conversational_category_examples(category: str, count: int) -> List[Dict[str, Any]]:
    """Create examples that mimic natural conversation patterns"""
    
    conversational_patterns = [
        "So I was talking to {person} about {topic}",
        "You know that {person} who {action}?",
        "Last weekend, {person} and I {activity}",
        "Did you hear that {person} {event}?",
        "My {relation} {person} just {action}"
    ]
    
    examples = []
    
    for i in range(count):
        pattern = random.choice(conversational_patterns)
        example = create_conversational_example(pattern, category)
        examples.append(example)
    
    return examples
```

#### Step 2: Evaluate Model
```python
def evaluate_model(model, tokenizer, eval_examples: List[Dict[str, Any]]):
    """Comprehensive model evaluation"""
    
    correct_relations = 0
    total_relations = 0
    correct_entities = 0
    total_entities = 0
    
    for example in eval_examples:
        # Get model prediction
        prediction = predict_relations(model, tokenizer, example["text"])
        
        # Evaluate entities
        predicted_entities = set([e["text"] for e in prediction["entities"]])
        actual_entities = set([e["text"] for e in example["entities"]])
        
        correct_entities += len(predicted_entities.intersection(actual_entities))
        total_entities += len(actual_entities)
        
        # Evaluate relations
        predicted_relations = set([
            f"{r['subject']}|{r['predicate']}|{r['object']}" 
            for r in prediction["relations"]
        ])
        actual_relations = set([
            f"{r['subject']}|{r['predicate']}|{r['object']}" 
            for r in example["relations"]
        ])
        
        correct_relations += len(predicted_relations.intersection(actual_relations))
        total_relations += len(actual_relations)
    
    entity_accuracy = correct_entities / total_entities if total_entities > 0 else 0
    relation_accuracy = correct_relations / total_relations if total_relations > 0 else 0
    
    return {
        "entity_accuracy": entity_accuracy,
        "relation_accuracy": relation_accuracy,
        "overall_accuracy": (entity_accuracy + relation_accuracy) / 2
    }
```

#### Step 3: Error Analysis
```python
def analyze_errors(model, tokenizer, eval_examples: List[Dict[str, Any]]):
    """Analyze model errors for targeted improvement"""
    
    errors = {
        "wrong_relation_type": [],
        "missing_entities": [],
        "extra_entities": [],
        "missing_relations": [],
        "extra_relations": []
    }
    
    for example in eval_examples:
        prediction = predict_relations(model, tokenizer, example["text"])
        
        # Analyze errors
        # ... detailed error analysis logic
        
    return errors
```

### Phase 4: Optimization and Deployment (Week 2-3)

#### Step 1: JSON Schema Enforcement
```python
import json
import jsonschema
from typing import Dict, Any, Optional

# Enhanced JSON Schema for conversational relation extraction
CONVERSATIONAL_EXTRACTION_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "type": {
                        "type": "string",
                        "enum": ["PERSON", "ORG", "ORGANIZATION", "LOC", "LOCATION", "PRODUCT", 
                                "MISC", "EVENT", "DATE", "TITLE", "NUMBER"]
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["text", "type", "confidence"]
            }
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "minLength": 1},
                    "predicate": {"type": "string", "minLength": 1},
                    "object": {"type": "string", "minLength": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["subject", "predicate", "object", "confidence"]
            }
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["entities", "relations", "confidence"]
}

def validate_and_fix_json(output_text: str) -> Dict[str, Any]:
    """Validate and fix JSON output from model"""
    
    # Try to extract JSON from model output
    json_start = output_text.find('{')
    json_end = output_text.rfind('}') + 1
    
    if json_start == -1 or json_end == -1:
        return create_empty_result()
    
    json_str = output_text[json_start:json_end]
    
    # Try to parse JSON
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common JSON errors
        fixed_json = fix_common_json_errors(json_str, e)
        try:
            result = json.loads(fixed_json)
        except json.JSONDecodeError:
            return create_empty_result()
    
    # Validate against schema
    try:
        jsonschema.validate(result, CONVERSATIONAL_EXTRACTION_SCHEMA)
    except jsonschema.ValidationError:
        # Try to fix schema violations
        result = fix_schema_violations(result)
        try:
            jsonschema.validate(result, CONVERSATIONAL_EXTRACTION_SCHEMA)
        except jsonschema.ValidationError:
            return create_empty_result()
    
    return result

def fix_common_json_errors(json_str: str, error: json.JSONDecodeError) -> str:
    """Fix common JSON parsing errors"""
    
    fixed = json_str
    
    # Fix missing quotes around property names
    fixed = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', fixed)
    
    # Fix trailing commas
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    
    # Fix single quotes instead of double quotes
    fixed = fixed.replace("'", '"')
    
    # Fix unescaped quotes in strings
    lines = fixed.split('\n')
    for i, line in enumerate(lines):
        if line.count('"') % 2 == 1 and i < len(lines) - 1:
            lines[i] = line + '"'
            lines[i + 1] = '"' + lines[i + 1]
    fixed = '\n'.join(lines)
    
    return fixed

def fix_schema_violations(result: Dict[str, Any]) -> Dict[str, Any]:
    """Fix common schema violations"""
    
    # Ensure required fields exist
    if "entities" not in result:
        result["entities"] = []
    if "relations" not in result:
        result["relations"] = []
    if "confidence" not in result:
        result["confidence"] = 0.0
    
    # Fix entity types
    valid_entity_types = ["PERSON", "ORG", "ORGANIZATION", "LOC", "LOCATION", 
                          "PRODUCT", "MISC", "EVENT", "DATE", "TITLE", "NUMBER"]
    
    for entity in result["entities"]:
        if "type" in entity and entity["type"] not in valid_entity_types:
            # Map common variations
            type_mapping = {
                "PERSONAL": "PERSON",
                "COMPANY": "ORG",
                "BUSINESS": "ORG",
                "PLACE": "LOCATION",
                "TIME": "DATE",
                "THING": "MISC"
            }
            entity["type"] = type_mapping.get(entity["type"], "MISC")
        
        # Ensure confidence is valid
        if "confidence" not in entity:
            entity["confidence"] = 0.5
        entity["confidence"] = max(0, min(1, entity["confidence"]))
    
    # Fix relations
    for relation in result["relations"]:
        # Ensure required fields
        for field in ["subject", "predicate", "object"]:
            if field not in relation:
                relation[field] = ""
        
        # Ensure confidence is valid
        if "confidence" not in relation:
            relation["confidence"] = 0.5
        relation["confidence"] = max(0, min(1, relation["confidence"]))
    
    # Fix overall confidence
    result["confidence"] = max(0, min(1, result["confidence"]))
    
    return result

def create_empty_result() -> Dict[str, Any]:
    """Create empty result when parsing fails"""
    return {
        "entities": [],
        "relations": [],
        "confidence": 0.0
    }
```

#### Step 2: Optimize for Inference
```python
from transformers import pipeline
import torch

# Create optimized pipeline
relation_extractor = pipeline(
    "text-generation",
    model="./relation-extractor-v3-final",
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16,
    max_length=512,
    temperature=0.1,
    do_sample=False
)
```

#### Step 2: Create Enhanced Inference Function
```python
def extract_relations_conversational(text: str) -> Dict[str, Any]:
    """Enhanced inference function for conversational relation extraction"""
    
    # Create conversational prompt
    prompt = f"""Extract entities and relations from this conversational text:

Text: {text}

Task: Identify relationships in natural conversation about everyday topics.
Focus on: family, friends, work, entertainment, sports, food, travel, and current events.

Output format:
{{"entities": [{{"text": "Entity Name", "type": "TYPE", "confidence": 0.9}}], "relations": [{{"subject": "Entity1", "predicate": "relation_type", "object": "Entity2", "confidence": 0.8}}], "confidence": 0.85}}

Important: Return only valid JSON with proper formatting."""
    
    # Generate with error handling
    try:
        result = relation_extractor(prompt, max_new_tokens=512, temperature=0.2)
        output = result[0]['generated_text']
        
        # Use enhanced JSON validation and fixing
        extraction = validate_and_fix_json(output)
        
        # Apply conversational post-processing
        extraction = apply_conversational_post_processing(extraction, text)
        
        return extraction
    
    except Exception as e:
        print(f"Inference error: {e}")
        return {"entities": [], "relations": [], "confidence": 0.0}

def apply_conversational_post_processing(extraction: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    """Apply post-processing rules specific to conversational text"""
    
    # Handle personal pronouns and references
    extraction = resolve_pronoun_references(extraction, original_text)
    
    # Improve relation type accuracy for conversational language
    extraction = refine_conversational_relations(extraction)
    
    # Adjust confidence based on conversational clarity
    extraction = adjust_conversational_confidence(extraction, original_text)
    
    return extraction

def resolve_pronoun_references(extraction: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Resolve personal pronouns to their referents"""
    
    # Simple pronoun resolution patterns
    pronoun_patterns = {
        "I": "speaker",
        "me": "speaker", 
        "my": "speaker",
        "we": "speaker_group",
        "us": "speaker_group",
        "our": "speaker_group"
    }
    
    for relation in extraction["relations"]:
        # Replace pronouns with resolved references
        for field in ["subject", "object"]:
            if relation[field] in pronoun_patterns:
                relation[field] = pronoun_patterns[relation[field]]
    
    return extraction

def refine_conversational_relations(extraction: Dict[str, Any]) -> Dict[str, Any]:
    """Refine relation types based on conversational context"""
    
    conversational_mappings = {
        # Common conversational patterns
        "works_for": ["employed by", "works at", "has job at"],
        "friends_with": ["friends with", "good friends with", "hangs out with"],
        "lives_in": ["lives in", "lives at", "stays in", "based in"],
        "studied_at": ["studied at", "went to", "attended"],
        "married_to": ["married to", "husband of", "wife of", "spouse of"]
    }
    
    for relation in extraction["relations"]:
        predicate = relation["predicate"].lower()
        
        # Check for conversational variations
        for standard_predicate, variations in conversational_mappings.items():
            if any(var in predicate for var in variations):
                relation["predicate"] = standard_predicate
                break
    
    return extraction

def adjust_conversational_confidence(extraction: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Adjust confidence based on conversational clarity"""
    
    # Factors that affect confidence in conversational text
    uncertainty_indicators = ["maybe", "might", "could be", "I think", "probably", "perhaps"]
    
    # Check for uncertainty indicators
    has_uncertainty = any(indicator in text.lower() for indicator in uncertainty_indicators)
    
    # Check for pronoun usage (reduces confidence)
    pronoun_count = sum(1 for word in text.split() if word.lower() in ["i", "me", "my", "we", "us", "our"])
    
    # Adjust overall confidence
    if has_uncertainty:
        extraction["confidence"] *= 0.8
    elif pronoun_count > 2:
        extraction["confidence"] *= 0.9
    
    # Ensure confidence stays within bounds
    extraction["confidence"] = max(0.1, min(1.0, extraction["confidence"]))
    
    return extraction
```

#### Step 3: Test Performance
```python
import time

def test_performance():
    """Test inference speed and accuracy"""
    
    test_texts = [
        "Tim Cook is the CEO of Apple.",
        "Sarah works at Google as a software engineer.",
        "Microsoft is headquartered in Redmond, Washington.",
        "Tesla develops electric vehicles.",
        "John and Mary are colleagues at Microsoft."
    ]
    
    times = []
    for text in test_texts:
        start_time = time.time()
        result = extract_relations(text)
        end_time = time.time()
        times.append(end_time - start_time)
        
        print(f"Text: {text}")
        print(f"Time: {end_time - start_time:.3f}s")
        print(f"Result: {result}")
        print()
    
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time:.3f}s")
```

## 🎯 Success Criteria

### Training Success
- [ ] 10K high-quality diverse conversational examples generated
- [ ] Model converges with low validation loss
- [ ] Entity accuracy > 90% on conversational test set
- [ ] Relation accuracy > 80% on conversational test set
- [ ] Balanced performance across all 6 conversation categories

### Performance Success
- [ ] Inference time < 0.5s on average
- [ ] Model size < 1GB for deployment
- [ ] Consistent performance across conversation categories
- [ ] No hallucination or false relations in conversational contexts

### Quality Success
- [ ] Correct relation types in >85% of conversational examples
- [ ] Proper entity boundaries in natural speech
- [ ] High confidence scores for correct predictions
- [ ] Low false positive rate in ambiguous contexts
- [ ] Handles conversational language and personal pronouns correctly

### Conversational Coverage
- [ ] Daily Life examples: >85% accuracy
- [ ] Work & Career examples: >85% accuracy
- [ ] Entertainment examples: >85% accuracy
- [ ] Sports & Hobbies examples: >85% accuracy
- [ ] Food & Travel examples: >85% accuracy
- [ ] News & Events examples: >85% accuracy

### Real-world Performance
- [ ] Handles personal pronouns (I, me, my, our) correctly
- [ ] Understands conversational context and references
- [ ] Extracts relations from natural speech patterns
- [ ] Works with first-person and third-person perspectives
- [ ] Processes temporal references (yesterday, last week) correctly

## 🚀 Deployment Checklist

### Model Deployment
- [ ] Model quantized for efficient inference
- [ ] LM Studio compatibility verified
- [ ] JSON schema validation working
- [ ] Error handling implemented

### Integration Testing
- [ ] Integration with HotMem V3 pipeline
- [ ] End-to-end testing with voice input
- [ ] Performance under load testing
- [ ] Edge case handling verified

### Production Ready
- [ ] Monitoring and logging in place
- [ ] Fallback mechanisms implemented
- [ ] Performance benchmarks documented
- [ ] User acceptance testing completed

---

This training guide provides a comprehensive approach to creating a production-ready relation extractor that maintains our elegant architecture while achieving 90%+ accuracy.