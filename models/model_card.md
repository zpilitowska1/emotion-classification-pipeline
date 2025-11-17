# Model Card: Polish Emotion Classification Model

## 1. Model Overview

**Architecture:**
This model is a fine-tuned **XLM-RoBERTa-large** transformer-based neural network designed for multi-label emotion classification in Polish text. The architecture consists of:

- **Base Model**: XLM-RoBERTa-large (559,897,607 parameters)
- **Classification Head**: Custom multi-label classifier with dropout regularization (0.1)
- **Output Layer**: 7 emotion categories using sigmoid activation 
- **Input Processing**: Maximum sequence length of 512 tokens
- **Training Strategy**: Mixed precision training with gradient accumulation (effective batch size: 32)

**Technical Specifications:**
- Model Type: Sequence Classification (Multi-label)
- Framework: PyTorch + Hugging Face Transformers
- Tokenizer: XLM-RoBERTa tokenizer with multilingual support
- Problem Type: Multi-label binary classification

**Design Choices:**
- **Why XLM-RoBERTa-large?** 

Chosen for its strong multilingual capabilities and proven performance on Polish language tasks. The large variant provides better representation learning for the nuanced emotion detection required.
- **Multi-label Classification**: 

Allows simultaneous detection of multiple emotions, reflecting real-world emotional complexity (e.g., "angry surprise" or "fearful sadness").
- **No Layer Freezing**: 

All transformer layers were trained (freeze_layers=0) to maximize adaptation to the specific Polish emotion detection task.
- **Simple Preprocessing**: 

Minimal text normalization to preserve emotional markers like punctuation, capitalization, and expressive spelling (e.g., "!!!!").

**Purpose:**

The model was developed to automatically detect and classify emotions in Polish-language media content, specifically targeting television transcripts from reality competition shows (e.g. MasterChef). Primary goals include:

1. Identify 6 core emotions (happiness, sadness, anger, surprise, fear, disgust) plus neutral state
2. Enable automated emotional content analysis for broadcast monitoring
3. Account for Polish-specific emotional expression patterns in informal spoken language
4. Achieve F1-score ≥ 0.75 per emotion for reliable production use

**Development Context:**

*Key Assumptions:*
- Twitter data (TwitterEmo dataset) provides reasonable proxy for informal spoken Polish despite domain mismatch
- Emotional expressions in social media and reality TV share linguistic characteristics (informality, expressiveness)
- Multi-label approach better captures emotional complexity than single-label classification
- XLM-RoBERTa's multilingual pre-training transfers well to Polish despite limited Polish-specific fine-tuning data

*Relevant Constraints:*
- **Data Scarcity**: Limited Polish emotion-labeled datasets, especially for target domain (TV transcripts)
- **Class Imbalance**: Training data heavily skewed toward neutral/negative emotions (fear: 0.9%, sadness: 4.6%)
- **Domain Mismatch**: Training on Twitter, deploying on TV transcripts (different vocabulary, context)
- **Computational Resources**: Single GPU training (NVIDIA RTX 2000 Ada, 8GB VRAM) limited batch size and model size exploration

*Development Conditions:*
- **Training Environment**: Single NVIDIA RTX 2000 Ada Generation GPU with 8GB VRAM
- **Training Duration**: Estimated 5,986 minutes (~100 hours) for 100 epochs at 59.9 min/epoch
- **Data Split**: 28,736 training samples, 7,185 validation samples (80/20 split)
- **Evaluation Set**: 791 held-out samples from actual TV transcript (group_17_url_1_transcript.csv)
- **Random Seed**: 42 (for reproducibility)

## 2. Intended Use


*Primary Use Cases:*
1. **Automated Emotion Detection in Television Transcripts**
   - Real-time analysis of reality TV competitions
   - Identifies emotional peaks/troughs in episodes for content editing and highlight reels
   - Example: Detecting fear reactions during time-pressure challenges, anger during judge critiques

2. **Content Moderation and Quality Control**
   - Flags emotionally intense segments for review before broadcast
   - Monitors emotional balance across episodes to ensure engaging content
   - Example: Ensuring appropriate emotional variety (not all negative) in episode structure

3. **Audience Engagement Analysis**
   - Correlates on-screen emotions with viewer engagement metrics
   - Identifies emotional patterns that resonate with audiences
   - Example: Determining which emotional arcs (fear→relief, frustration→triumph) drive viewership



**Limitations and Contexts Where Model Should NOT Be Used:**

*Technical Limitations:*
1. **Low-Resource Emotion Classes**
   - **Fear (F1: 0.404) and Sadness (F1: 0.490)**: Performance below acceptable threshold
   - **DO NOT USE** for applications requiring reliable fear/sadness detection (e.g., mental health monitoring)
   - Underrepresented in training data 

2. **Single-Emotion Dependency**
   - Model shows extreme brittleness: confidence crashes after removing 1-3 tokens
   - **DO NOT USE** for adversarial contexts where malicious users might manipulate input
   - Vulnerable to typos, paraphrasing, spelling variations

3. **Punctuation-Driven Classifications**
   - Happiness detection focuses on punctuation ("!") rather than emotion words
   - **DO NOT USE** for formal text analysis (news articles, legal documents) where punctuation usage differs

*Contextual Limitations:*
1. **Domain Specificity**
   - Trained on informal social media → Deployed on TV transcripts
   - **Best for**: Informal, competitive, high-stress conversational contexts
   - **Not suitable for**: Formal interviews, scripted drama, educational content

2. **Language and Cultural Context**
   - Optimized for Polish linguistic patterns and cultural emotional expression
   - **DO NOT USE** for other languages without retraining


**Connection to Client Needs (For Media Production Companies):*
- **Real-time Emotion Tracking**: Provides instant emotional content analysis during post-production editing
- **Cost Reduction**: Automates labor-intensive manual emotion tagging 
- **Content Optimization**: Identifies emotionally engaging moments for social media clips and promotional materials
- **Competitive Intelligence**: Analyzes competitor shows' emotional strategies


## 3. Dataset Details

**Training Data:**

*Primary Source: Polish TwitterEmo Dataset*
- **Source**: clarin-pl/twitteremo (Hugging Face Datasets)
- **Size**: 35,921 Polish tweets
- **Characteristics**: Informal, emotionally expressive social media content
- **Annotations**: Multi-label emotion tags across 13 categories (including sentiment and complex emotions)

*Training Split:*
- **Training Set**: 28,736 examples (80%)
- **Validation Set**: 7,185 examples (20%)
- **Random Seed**: 42 (stratified split by emotion distribution)

**Evaluation Data:**

*Held-Out Test Set:*
- **Source**: group_17_url_1_transcript.csv (Polish Hell's Kitchen/MasterChef transcript)
- **Size**: 791 sentences
- **Domain**: Reality TV competition dialogue
- **Annotations**: Manual emotion labels 
- **Distribution**: 
  - Neutral: 413 (52.2%)
  - Happiness: 153 (19.3%)
  - Surprise: 67 (8.5%)
  - Fear: 58 (7.3%)
  - Sadness: 56 (7.1%)
  - Anger: 35 (4.4%)
  - Disgust: 9 (1.1%)

**Preprocessing Steps:**

*Text Normalization (Simple Pipeline):*
1. **Twitter-Specific Cleaning**:
   - URLs → `[URL]`
   - Mentions (@username) → `[MENTION]`
   - Hashtags (#tag) → `[HASHTAG] tag`
   - Retweets (RT @...) → `[RETWEET]`

2. **Punctuation Normalization**:
   - Quotation marks standardized to ASCII
   - Excessive punctuation reduced (e.g., "!!!!" → "!!!")
   - Whitespace normalized (multiple spaces → single space)

3. **Tokenization**:
   - XLM-RoBERTa tokenizer with 512 max tokens
   - Automatic padding and truncation
   - Preserved all special tokens

**Distribution Details:**

*Emotion Class Distribution (Training Set):*
```
Emotion          Count      Percentage   Balance Issue
---------------------------------------------------------
neutral          14,821     51.6%        OVERREPRESENTED
disgust           6,673     23.2%        
anger             5,083     17.7%        
happiness         3,316     11.5%        
surprise          1,873      6.5%        
sadness           1,325      4.6%        UNDERREPRESENTED
fear                255      0.9%        SEVERELY UNDERREPRESENTED
---------------------------------------------------------

```

### Representativeness Across Languages and Cultures

**Language Diversity:**

*Current State:*
- **Monolingual**: Polish-only training and evaluation data
- **Dialect Coverage**: Primarily standard Polish from social media 
- **Register**: Informal, conversational Polish (tweets, spoken dialogue)

*Multilingual Challenges Addressed:*
1. **Tokenization**: XLM-RoBERTa handles Polish diacritics correctly (ą, ć, ę, etc.)
2. **Cultural Expression**: Model trained on Polish-specific emotional expression patterns
3. **Transfer Learning**: Base model pre-trained on 100 languages provides cross-lingual features

*Known Limitations:*
- Not representative of formal Polish (news, academic, legal text)
- Dataset primarily from Polish Twitter users (urban, younger demographic)
- Domain vocabulary coverage: Only 2.1% cooking/competition terms

**Cultural Considerations:**

*Emotional Expression Patterns:*
- Model learns Polish-specific emotional markers (vocabulary, punctuation, spelling)
- Twitter data may not fully represent emotional expression in competitive TV contexts
- 
*Bias Potential:*
- Social media skew: May overrepresent certain demographic/cultural expressions
- Age bias: Twitter users skew younger
- Platform bias: Twitter emotional expression ≠ real-world conversation

## 4. Performance Metrics and Evaluation

Error analysis file link: ........

### Overall Performance (791 TV transcript samples)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Macro-F1** | 0.385 | Poor multi-emotion capability |
| **Weighted-F1** | 0.617 | Inflated by neutral majority |
| **Accuracy** | 61.69% | Misleading - hides per-emotion failures |

### Per-Emotion Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Anger | 0.000 | 0.000 | 0.000 | 35 |
| Fear | 1.000 | 0.086 | 0.159 | 58 |
| Surprise | 0.433 | 0.194 | 0.268 | 67 |
| Sadness | 0.696 | 0.286 | 0.405 | 56 |
| Happiness | 0.853 | 0.418 | 0.561 | 153 |
| Disgust | 0.556 | 0.556 | 0.556 | 9 |
| Neutral | 0.618 | 0.956 | 0.751 | 413 |

### Critical Findings from Error Analysis

**1. Neutral Collapse (64% of errors)**
- 64% of emotional content misclassified as neutral
- Root cause: 52% neutral majority + unweighted loss function
- Result: Model defaults to neutral when uncertain

**2. Complete Emotion Detection Failures**
- Anger: 0% recall (complete failure - cannot detect at all)
- Fear: 8.6% recall (near-complete failure)
- Surprise: 19.4% recall (severe under-performance)
- Sadness: 28.6% recall (severe under-performance)

**Root causes:**
- Insufficient training data (fear: 0.9% of training set, anger: 17.7% but still failed)
- Extreme class imbalance not addressed by loss function
- Domain mismatch between Twitter training and TV transcript evaluation

**3. Sentence Complexity Effect**
- Misclassified sentences 30% longer (38 vs 29 characters)
- Complex, multi-clause emotional expressions systematically missed

**4. Domain Mismatch Impact**
- Training: Explicit Twitter expressions ("I'm SO ANGRY!! 😠")
- Evaluation: Implicit conversational cues ("What are you waiting for?")
- Model cannot recognize subtle, context-dependent emotions

### Production Readiness Assessment

**Acceptable Uses:**
- Neutral detection (95.6% recall)
- Binary emotion/neutral classification (~70% accuracy)
- Aggregate episode-level statistics

**NOT Recommended:**
- Individual emotion detection (macro-F1 too low)
- Anger detection (complete failure)
- Fear/sadness detection (unreliable)
- Safety-critical applications

  
## 5. Explainability and Transparency

Three complementary explainability techniques were applied to 21 high-confidence predictions:

1. Gradient × Input Attribution
2. Layer-wise Relevance Propagation
3. Input Perturbation Analysis
   
**Key Findings:**

*Attribution Patterns by Emotion:*

**Fear (Best Attribution Alignment):**
- Correctly focuses on emotion words: "obawiam" (fear), "martwię" (worry)
- Attribution scores: 1.2-2.5 (highest across all emotions)
- Example: "A ja zaczynam się martwić" → "martwię" gets 2.5 attribution
- **Interpretation**: Model learns fear through specific emotion vocabulary
- **Limitation**: Low recall (0.294) due to insufficient training examples

**Happiness (Problematic Attribution):**
- IGNORES emotion verbs: "uwielbiam" (love) gets near-zero attribution
- Punctuation dependency: "!" gets highest attribution in "Jesteście super!"
- **Interpretation**: Model uses lexical shortcuts instead of emotion semantics

**Disgust (Good Performance):**
- Appropriate emotion word attribution
- High F1 (0.707) correlates with 6,673 training examples
- **Interpretation**: Sufficient data enables proper pattern learning

**Neutral (Best Overall):**
-  Highest F1 score (0.817)
- Learns absence of emotion markers rather than specific patterns
- Benefits from being most abundant class (51.6%)

**Interpretation**: Model uses fundamentally different strategies:
- **High scale (fear, disgust)**: Keyword-driven, brittle but confident
- **Low scale (happiness, surprise)**: Context-driven, flexible but error-prone



**Decision-Making Transparency:**

*What the model sees:*
- **Input**: Polish sentence tokenized into max 512 subword tokens
- **Processing**: 24 transformer layers extract contextualized representations
- **Output**: 7 emotion probability scores (0.0-1.0) via sigmoid activation

*How it decides:*
1. **Feature extraction**: XLM-RoBERTa encodes semantic and syntactic features
2. **Contextualization**: Each token representation influenced by surrounding context (512-token window)
3. **Pooling**: [CLS] token aggregates sentence-level representation
4. **Classification**: Linear layer + sigmoid produces per-emotion probabilities
5. **Thresholding**: Probability > 0.5 → Emotion predicted 


## 6. Recommendations for Use

### Practical Deployment Guidance

**Operational Deployment:**

*For Media Production Companies:*

**Primary Use Case: Post-Production Emotion Tagging**
```python
# Recommended deployment workflow
import torch
from transformers import AutoTokenizer

# Load model
model, tokenizer, labels = load_polish_emotion_model("polish_emotion_model.h5")
model.eval()

# Process TV transcript segment
transcript = "Jesteście super! Ale martwię się o ten sos."
inputs = tokenizer(transcript, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

# Get predictions
with torch.no_grad():
    logits = model(inputs['input_ids'], inputs['attention_mask'])
    probabilities = torch.sigmoid(logits)

# Apply confidence threshold (CRITICAL for low-performing emotions)
CONFIDENCE_THRESHOLD = 0.7  # Only accept high-confidence predictions
predictions = (probabilities > CONFIDENCE_THRESHOLD).int()

# Map to emotions
detected_emotions = [labels[i] for i, pred in enumerate(predictions[0]) if pred == 1]

# RECOMMENDED: Manual review for fear/sadness
if 'fear' in detected_emotions or 'sadness' in detected_emotions:
    print(f"âš ï¸  MANUAL REVIEW REQUIRED: Low-confidence emotion detected")
    print(f"   Confidence: fear={probabilities[0][labels.index('fear')]:.2f}, sadness={probabilities[0][labels.index('sadness')]:.2f}")
```

**Operational Risks and Mitigations:**

| Risk | Mitigation Strategy |
|------|---------------------|
| **False positives** (happiness from "!") | Manually verify happiness predictions with multiple "!"<br>Check for emotion verbs in context<br>Use precision threshold >0.7 |
| **Missed fear/sadness** (low recall) | Always combine with keyword search for fear/sad vocabulary<br>Escalate uncertain cases to human review<br>Consider rule-based fallback for these emotions |
| **Brittle predictions** (1-3 token dependency) | Do NOT use for adversarial contexts<br>Avoid using on heavily edited/paraphrased text<br>Validate on diverse phrasings during testing |
| **Domain mismatch** (Twitter → TV) | Fine-tune on target domain data before deployment<br>Monitor performance on cooking/competition vocabulary<br>Supplement with domain-specific lexicons |
| **Computational cost** (560M parameters) | Use batch inference (16-32 samples)<br>Deploy on GPU infrastructure<br>Cache predictions for frequently analyzed content |

**Use Case Scenarios:**

** RECOMMENDED Use Cases:**

1. **Highlight Reel Generation**
   - Use **neutral, disgust, happiness** detections (F1 > 0.69)
   - Identify emotionally intense moments for promotional clips

2. **Content Moderation**
   - Flag segments with high anger/disgust for human review
   - Example: Detect contestant arguments (anger) for content rating assessment

3. **Emotional Arc Analysis**
   - Track emotional trajectory across full episodes
   - Identify neutral→emotion transitions
   - Example: Map contestant emotional journey (confidence → fear → relief)

4. **Competitor Analysis**
   - Benchmark emotional content against competitor shows
   - Use aggregate statistics (emotion distribution per episode)
   - Example: Compare Hell's Kitchen PL vs. international versions

** NOT RECOMMENDED Use Cases:**

1. **Mental Health Assessment**
   - Fear/sadness detection too unreliable (F1: 0.404, 0.490)
   - Model not trained for clinical contexts
2. **Real-time Live Broadcast Decisions**
   - Brittleness makes it unsuitable for adversarial or unpredictable contexts
   - Single-token dependency creates false trigger risk

3. **Legal/Compliance Decisions**
   - Model errors (especially false negatives) could have legal implications
   - Not validated for regulatory compliance

4. **Non-Polish Content**
   - Model specifically optimized for Polish language patterns
   - No validation on other languages


## 7. Sustainability Considerations

**Training Carbon Footprint:**

*Computational Requirements:*
- **Hardware**: Single NVIDIA RTX 2000 Ada Generation Laptop GPU (8GB VRAM)
- **Training Duration**: Estimated 100 hours for 100 epochs (59.9 min/epoch)
- **Model Size**: 560M parameters (~2.09 GB in FP32)

*Energy Consumption Estimates:*
```
Component                Value              Calculation
-----------------------------------------------------------------------
GPU TDP                  60W                NVIDIA RTX 2000 Ada spec
Training Time            100 hours          100 epochs × 59.9 min/epoch
Total GPU Energy         6.0 kWh            60W × 100h
System Energy (1.5×)     9.0 kWh            GPU + CPU + memory overhead
-----------------------------------------------------------------------
```

*Carbon Emissions (assuming grid mix):*
```
Region                   CO₂ per kWh    Total Training Emissions
-----------------------------------------------------------------------
Global Average           0.475 kg       4.28 kg CO₂
EU Average               0.295 kg       2.66 kg CO₂  
Poland (coal-heavy)      0.766 kg       6.89 kg CO₂
Renewable Energy         0.02 kg        0.18 kg CO₂
-----------------------------------------------------------------------
Equivalent to: ~30 km car travel (EU average electricity)
```

**Deployment Environmental Impact:**

*Inference Costs:*
- **Per-sentence inference**: ~128 ms (NVIDIA RTX 2000 Ada)
- **Power consumption**: ~60W GPU + 15W system = 75W total
- **Energy per 1000 sentences**: ~2.7 Wh

*Scaling Estimates:*
```
Usage Scenario                 Sentences/Day    Energy/Day    CO₂/Year (EU)
-------------------------------------------------------------------------
Small production (1 show)       10,000           27 Wh         2.9 kg
Medium production (5 shows)     50,000           135 Wh        14.5 kg  
Large production (20 shows)     200,000          540 Wh        58.1 kg
-------------------------------------------------------------------------
```


**Environmental Optimization Recommendations:**

*Training Phase:*
1. Use Efficient Hardware
2. **Optimize Training Strategy**:
   - Mixed precision training (FP16) reduces compute time ~40%
   - Early stopping (patience: 10) prevents wasted compute
   - Recommendation: Use gradient checkpointing for larger models (trade compute for memory)

3. **Transfer Learning Benefits**:
   - Fine-tuning pre-trained XLM-RoBERTa avoids training from scratch
   - Estimate: Pre-training 560M parameters would require ~10,000× more compute
   - **Carbon saved**: ~42,800 kg CO₂ (vs. training from scratch)


*Deployment Phase:*
1. **Batch Processing**:
   - Already uses batching (batch_size=16)
   - Process offline during low-energy hours

2. **Model Compression** (Future Work):
   - **Quantization**: Convert FP32 → INT8 (4× size reduction, 2-3× speedup)
   - **Knowledge Distillation**: Train smaller student model (XLM-RoBERTa-base: 279M params)
     - Estimated energy savings: 50% reduction
     - Trade-off: ~5% F1-score drop acceptable for high-volume inference
   - **Pruning**: Remove less important parameters (10-30% size reduction possible)

**Sustainability Best Practices:**

*For Model Development:*
1. **Set clear success criteria** BEFORE training (F1 ≥ 0.75) to avoid unnecessary iterations
2. **Use small-scale experiments** (10% data, 3 epochs) to validate approach before full training
3. **Monitor convergence closely**: Stop training when validation plateaus (early stopping implemented)
4. **Reuse validated architectures**: XLM-RoBERTa already proven for Polish emotion tasks

*For Model Deployment:*
1. **Cache predictions** for repeated content (e.g., re-analyses of same episodes)
2. **Prioritize high-value use cases**: Focus compute on most impactful applications
3. **Regular audits**: Monitor inference volume and energy consumption
4. **Upgrade triggers**: Only retrain when F1-score drops >5% or new domain requirements emerge

---
