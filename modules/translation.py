"""
Polish to English translation using Helsinki-NLP Marian MT
Based on machine_translation_best_model.ipynb
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from config import TRANSLATION_CONFIG

class MarianTranslator:
    """
    Polish to English translator using Marian MT
    Model: Helsinki-NLP/opus-mt-pl-en
    """
    
    def __init__(self, model_name=None):
        """Initialize translator with pre-trained model"""
        if model_name is None:
            model_name = TRANSLATION_CONFIG['model']
        
        print(f"    Loading translation model: {model_name}")
        
        # Load model and tokenizer
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f"    Translation model loaded on {self.device}")
    
    def translate_batch(self, texts, max_length=512):
        """
        Translate a batch of Polish texts to English
        
        Args:
            texts: List of Polish sentences
            max_length: Maximum sequence length
            
        Returns:
            List of English translations
        """
        if not texts:
            return []
        
        # Tokenize input (Polish)
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translations
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        
        # Decode to English text
        translations = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return translations

def translate_segments(segments):
    """
    Translate all Polish segments to English
    
    Args:
        segments: List of dicts with 'text' key containing Polish sentences
        
    Returns:
        List of English translations (same order as input)
    """
    print(f"[3/4] Translating {len(segments)} segments with Marian MT...")
    
    # Initialize translator
    translator = MarianTranslator()
    
    batch_size = TRANSLATION_CONFIG['batch_size']
    max_length = TRANSLATION_CONFIG['max_length']
    
    all_translations = []
    texts = [seg['text'] for seg in segments]
    
    # Process in batches for efficiency
    for i in tqdm(range(0, len(texts), batch_size), desc="    Translating"):
        batch = texts[i:i+batch_size]
        
        try:
            # Translate batch
            translations = translator.translate_batch(batch, max_length)
            all_translations.extend(translations)
            
        except Exception as e:
            print(f"    WARNING: Batch translation failed - {e}")
            
            # Fallback: translate one by one
            for text in batch:
                try:
                    trans = translator.translate_batch([text], max_length)[0]
                    all_translations.append(trans)
                except Exception as e2:
                    print(f"    ERROR translating '{text[:30]}...' - using original")
                    all_translations.append(text)
    
    print(f"    Translation completed")
    
    return all_translations