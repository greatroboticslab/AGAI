# Improving Hallucination Evaluation Metrics

## Current Metrics and Their Limitations

### 1. Keyword-Based Wrong-Disease Detection
**Current:** Checks if unique disease keywords appear in wrong context
**Limitation:** Too rigid, false positives from negations ("not waterlogged")
**Improvement:** ✅ Added negation detection, refined unique indicators

### 2. Keyword-Based Grounding
**Current:** Checks if expected symptom keywords appear
**Limitation:** Misses paraphrases ("wilted" vs "drooping")
**Score:** 59% (keyword-based)

### 3. Semantic Grounding (NEW)
**Current:** Uses sentence embeddings to find semantic matches
**Benefit:** Catches paraphrases, different terminology
**Score:** 18.7% - lower but more accurate

### 4. Training Alignment (NEW)
**Current:** Compares output to training data using embeddings
**Benefit:** Shows if model has drifted from training
**Score:** 63.4%

---

## How to Make Training JSON Align with Model Output

### Problem
The training data describes symptoms one way, but the model might generate descriptions differently.

### Solutions

#### Option 1: Update Knowledge Base to Match Model
After running semantic evaluation, extract common phrases from model outputs:

```python
# Extract most frequent symptom phrases from model outputs
from collections import Counter
import re

phrases = []
for result in semantic_results:
    text = result["generated_text"].lower()
    # Extract symptom-like phrases
    matches = re.findall(r'(leaves? (?:are|appear|show|have) \w+)', text)
    phrases.extend(matches)

common_phrases = Counter(phrases).most_common(20)
# Add these to knowledge_base.json
```

#### Option 2: Train a Symptom Paraphrase Model
Use the training data to learn what symptoms look like:

```python
# Collect all symptom descriptions from training data
symptom_sentences = []
for response in training_responses:
    # Split into sentences mentioning visual cues
    for sent in response.split('.'):
        if any(kw in sent.lower() for kw in ['leaf', 'stem', 'root', 'color', 'spot']):
            symptom_sentences.append(sent)

# Use these to build a symptom-specific embedding model
```

#### Option 3: LLM-as-Judge (Best but Requires API)
Use GPT-4 or Claude to evaluate responses:

```python
def llm_judge_hallucination(response, disease, image_description=None):
    prompt = f"""
    You are evaluating a plant diagnostic response for hallucinations.
    
    Disease: {disease}
    Response: {response}
    
    Check for:
    1. Does it claim invisible things? (smells, underground parts if not shown)
    2. Does it describe wrong disease symptoms?
    3. Are treatments appropriate for this disease?
    
    Return JSON: {{"score": 0.0-1.0, "issues": ["list"]}}
    """
    # Call LLM API
    return llm_response
```

---

## Making Grounding Work Beyond Keyword Matching

### Current Approach
```
Expected: "gray fuzzy growth"
Generated: "fuzzy gray coating"
Keyword match: FAIL (different word order)
```

### Semantic Approach (Implemented)
```
Expected: "gray fuzzy growth"  → embedding [0.1, 0.3, ...]
Generated: "fuzzy gray coating" → embedding [0.12, 0.28, ...]
Cosine similarity: 0.89 → MATCH!
```

### Advanced: Entity-Based Grounding
Extract plant entities and verify them:

```python
# Use spaCy or custom NER to extract:
# - Plant parts mentioned: leaves, stems, roots
# - Conditions described: wilted, discolored, fuzzy
# - Colors: brown, gray, yellow

# Then verify:
# 1. Are mentioned plant parts visible in image?
# 2. Are conditions appropriate for this disease?
# 3. Are colors described accurately?
```

### Best Practice: Human-in-the-Loop Calibration

1. **Annotate 30-50 samples manually**
   - Is the response accurate? (1-5 scale)
   - Any wrong disease symptoms? (yes/no)
   - Any invisible claims? (yes/no)

2. **Use annotations to calibrate automatic metrics**
   - Find threshold where semantic score predicts human judgment
   - Adjust embedding model or similarity threshold

3. **Create "gold standard" test set**
   - 10-20 images with manually verified "correct" responses
   - Use for regression testing after model changes

---

## Quick Wins

### 1. Run Semantic Evaluation Regularly
```bash
# After each eval, run semantic analysis
python evaluate_semantic.py --results results/latest.json
```

### 2. Compare Training Alignment Over Time
```bash
# Track if model drifts from training
python -c "
from evaluate_semantic import evaluate_existing_results
# Compare alignment scores across evaluations
"
```

### 3. Flag Low-Confidence Responses
```python
# In demo_v5.py, warn user if ResNet confidence < 70%
if confidence < 0.7:
    response += "\n\n⚠️ *Low confidence diagnosis. Consider expert review.*"
```

### 4. Add Symptom Counter to Output
```python
# Show which expected symptoms were found
matched = ["wilting (✓)", "brown edges (✓)", "fuzzy growth (✗)"]
response += f"\n\n**Symptoms detected:** {', '.join(matched)}"
```

---

## Recommended Evaluation Pipeline

```
1. Run keyword-based evaluation (fast, catches obvious issues)
   └── python evaluate.py --num-samples 20

2. Run semantic evaluation (slower, catches paraphrases)
   └── python evaluate_semantic.py --results results/latest.json

3. Manual review of flagged responses (highest accuracy)
   └── Focus on: low alignment, low grounding, any hallucination flags

4. Update knowledge base with common model phrases
   └── Improves future keyword matching
```

