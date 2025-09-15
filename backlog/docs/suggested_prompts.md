# Suggested Prompts for Tier 2 & 3 Optimization

## Tier 2 (qwen3-0.6b-mlx) - Stop Repetition Issue

### Problem
qwen3 starts correctly but then repeats the extraction process instead of stopping.

### Solution Approaches

#### 1. **Minimal Direct Prompt** (Recommended)
```
Text: {text}

Extract as JSON:
```

**Parameters:**
- `max_tokens`: 150 (prevent repetition)  
- `stop`: ["\n\n", "Text:", "Extract:"]
- `temperature`: 0.0 (more deterministic)

#### 2. **One-Shot with Clear Boundary**
```
Extract facts as JSON. Example:

Text: "John works at Apple"
{"entities": [{"name": "John", "type": "Person"}], "relationships": [{"source": "John", "relation": "works_at", "target": "Apple"}]}

Text: {text}
```

**Parameters:**
- `max_tokens`: 200
- `stop`: ["}]\n", "Text:", "Example:"]
- `temperature`: 0.1

#### 3. **Instruction + Format**
```
Extract people, companies, products and their relationships from: {text}

JSON format only:
```

**Parameters:**
- `max_tokens`: 180
- `stop`: ["}\n\n", "Extract:", "JSON:"]
- `temperature`: 0.1

## Tier 3 (llama-3.2-1b-instruct) - Output Format

### Current Status
Works well but markdown parsing could be improved OR switch to JSON.

### Option A: Improve Markdown (Recommended)
```
Extract key facts from: {text}

Output exactly this format:

ENTITIES:
- Alice (Person)
- Tesla (Company)

RELATIONSHIPS:  
- Alice -> works_at -> Tesla
- Alice -> drives -> Model 3

Use this exact format only.
```

### Option B: Switch to JSON
```
Extract facts from: {text}

Return valid JSON only:
{"entities": [{"name": "Alice", "type": "Person"}], "relationships": [{"source": "Alice", "relation": "works_at", "target": "Tesla"}]}
```

## Testing Strategy

1. **Use the iteration script** to test these prompts quickly
2. **Focus on stop sequences** for Tier 2 to prevent repetition
3. **Measure consistency** - run same prompt 5 times
4. **Check JSON parsing** success rate

## Performance Targets

- **Tier 2**: <200ms, consistent JSON output
- **Tier 3**: <500ms, easily parseable format
- **Success rate**: >90% valid extractions

## Quick Commands

```bash
# Test specific prompt
python prompt_iteration_test.py
# Choose option 3 (quick test)
```