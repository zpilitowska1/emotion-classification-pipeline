"""
Emotion classification - Using CPUUnpickler for reliable CPU loading
Based on load_model.py approach
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
import pickle
import io
import os
import sys
from tqdm import tqdm
from config import EMOTION_CONFIG, EMOTION_LABELS

class PolishEmotionClassifier(nn.Module):
    """Custom emotion classifier matching training architecture"""
    
    def __init__(self, base_model, num_labels=7, dropout_rate=0.1):
        super(PolishEmotionClassifier, self).__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.config = base_model.config
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
    
    def get_input_embeddings(self):
        """Return input embeddings layer"""
        return self.base_model.roberta.embeddings.word_embeddings
    
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        """Forward pass - handles both input_ids and embeddings"""
        if inputs_embeds is not None:
            outputs = self.base_model.roberta(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
        else:
            outputs = self.base_model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get pooled output (compatible with different transformer versions)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class CPUUnpickler(pickle.Unpickler):
    """
    Custom unpickler that forces all tensors to CPU
    This fixes GPU→CPU loading issues
    """
    
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            # Force CPU loading for all tensors
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)

def load_emotion_model():
    """
    Load trained emotion model using CPUUnpickler
    Works on CPU-only machines even if model was trained on GPU
    """
    model_path = EMOTION_CONFIG['model_path']
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    
    print(f"    Loading model from: {model_path}")
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    
    try:
        # Load using custom unpickler (forces CPU)
        with open(model_path, 'rb') as f:
            save_dict = CPUUnpickler(f).load()
        
        print("    ✓ Model file loaded to CPU")
        
        # Extract configuration
        arch_config = save_dict['model_architecture']
        base_model_name = arch_config['base_model_name']
        num_labels = arch_config['num_labels']
        
        print(f"    Architecture: {arch_config.get('model_type', 'PolishEmotionClassifier')}")
        print(f"    Labels: {num_labels}")
        
        # Recreate base model
        print("    Loading XLM-RoBERTa base...")
        base_model = XLMRobertaForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        
        # Recreate classifier
        print("    Recreating classifier...")
        model = PolishEmotionClassifier(
            base_model=base_model,
            num_labels=num_labels,
            dropout_rate=arch_config.get('dropout_rate', 0.1)
        )
        
        # Load trained weights
        print("    Loading trained weights...")
        model.load_state_dict(save_dict['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load tokenizer
        print("    Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        print("    ✓ Model ready for inference")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def classify_emotions(texts):
    """
    Classify emotions for batch of texts
    
    Args:
        texts: List of Polish sentences
        
    Returns:
        List of emotion labels (comma-separated if multiple)
    """
    print(f"[4/4] Classifying emotions for {len(texts)} segments...")
    
    # Load model
    model, tokenizer, device = load_emotion_model()
    
    batch_size = EMOTION_CONFIG['batch_size']
    max_length = EMOTION_CONFIG['max_length']
    thresholds = EMOTION_CONFIG['thresholds']
    
    all_predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="    Classifying"):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            encoding = tokenizer(
                batch,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Get predictions
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Apply thresholds
            for prob_array in probs:
                detected = []
                
                for emotion, prob in zip(EMOTION_LABELS, prob_array):
                    threshold = thresholds.get(emotion, 0.5)
                    if prob >= threshold:
                        detected.append(emotion)
                
                # Default to neutral
                if not detected:
                    detected = ['neutral']
                
                all_predictions.append(','.join(detected))
    
    print(f"    ✓ Classification completed")
    
    return all_predictions