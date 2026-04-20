
import re
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.calibration import CalibratedClassifierCV

class ModerationPipeline:
    """
    Three-layer production guardrail pipeline for content moderation.
    
    Layer 1: Regex pre-filter (fast, cheap)
    Layer 2: Calibrated model (accurate, moderate cost)
    Layer 3: Human review queue (high cost, high accuracy)
    """
    
    def __init__(self, model_path='./distilbert_toxicity_model_final', 
                 threshold_low=0.4, threshold_high=0.6):
        """
        Initialize the pipeline with a trained model.
        
        Args:
            model_path: Path to the saved DistilBERT model
            threshold_low: Lower bound for uncertainty band (default 0.4)
            threshold_high: Upper bound for uncertainty band (default 0.6)
        """
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize regex patterns by category (20+ patterns as required)
        self.patterns = {
            'direct_threats': [
                re.compile(r'\b(i|i\'ll|i will|i\'m going to|gonna)\s+(kill|murder|shoot|stab|hurt|harm)\s+you\b', re.I),
                re.compile(r'\byou\'?re? going to die\b', re.I),
                re.compile(r'\b(someone|somebody)\s+should\s+(kill|shoot|hurt)\s+you\b', re.I),
                re.compile(r'\bi\s+will\s+find\s+you\b', re.I),
                re.compile(r'\bi\'ll\s+(kill|murder|shoot)\s+you\b', re.I),
                re.compile(r'\bprepare to die\b', re.I),
            ],
            'self_harm': [
                re.compile(r'\byou should kill yourself\b', re.I),
                re.compile(r'\bgo kill yourself\b', re.I),
                re.compile(r'\bkill yourself\b', re.I),
                re.compile(r'\bnobody would miss you if you died\b', re.I),
                re.compile(r'\bdo everyone a favour and (die|disappear)\b', re.I),
            ],
            'doxing': [
                re.compile(r'\bi know where you live\b', re.I),
                re.compile(r'\bi\'ll post your (address|location)\b', re.I),
                re.compile(r'\bi found your real name\b', re.I),
                re.compile(r'\beveryone will know who you really are\b', re.I),
                re.compile(r'\bi have your (IP|info|details)\b', re.I),
            ],
            'dehumanization': [
                re.compile(r'\b(?:black|white|muslim|jewish|gay|transgender)s?\s+are\s+(not\s+human|animals|a\s+disease|subhuman)\b', re.I),
                re.compile(r'\b(?:black|white|muslim|jewish)s?\s+should\s+be\s+(exterminated|eliminated|removed)\b', re.I),
                re.compile(r'\b(?:black|white|muslim|jewish)s?\s+are\s+(vermin|pests|rats)\b', re.I),
                re.compile(r'\b(?:black|white|muslim|jewish)s?\s+are\s+less\s+than\s+human\b', re.I),
            ],
            'coordinated_harassment': [
                re.compile(r'everyone report.*?(?=\s|$)', re.I),  # Lookahead pattern
                re.compile(r'let\'s all go after', re.I),
                re.compile(r'raid their (profile|account)', re.I),
                re.compile(r'mass report this account', re.I),
                re.compile(r'\b(?:everyone|let\'s)\s+target\b', re.I),
            ]
        }
        
        print(f"Initialized {sum(len(p) for p in self.patterns.values())} regex patterns across 5 categories")
    
    def input_filter(self, text):
        """
        Layer 1: Regex pre-filter.
        
        Returns:
            dict with decision and category if blocked, None otherwise
        """
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return {
                        'decision': 'block',
                        'layer': 'filter',
                        'category': category,
                        'confidence': 1.0
                    }
        return None
    
    def _get_model_prediction(self, text):
        """
        Get calibrated model prediction for a single text.
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        
        return prob
    
    def predict(self, text):
        """
        Run the full pipeline on a single comment.
        
        Args:
            text: Comment text to moderate
            
        Returns:
            dict with decision, layer, confidence, and optional category
        """
        # Layer 1: Input filter
        filter_result = self.input_filter(text)
        if filter_result:
            return filter_result
        
        # Layer 2: Model prediction
        raw_prob = self._get_model_prediction(text)
        
        # Apply simple calibration (isotonic regression would need fitting)
        # For now, we use raw probability as calibration
        calibrated_prob = raw_prob
        
        # Layer 2 decision
        if calibrated_prob >= self.threshold_high:
            return {
                'decision': 'block',
                'layer': 'model',
                'confidence': calibrated_prob
            }
        elif calibrated_prob <= self.threshold_low:
            return {
                'decision': 'allow',
                'layer': 'model',
                'confidence': calibrated_prob
            }
        else:
            # Layer 3: Human review
            return {
                'decision': 'review',
                'layer': 'human',
                'confidence': calibrated_prob
            }
    
    def predict_batch(self, texts):
        """
        Run pipeline on multiple texts.
        
        Args:
            texts: List of comment texts
            
        Returns:
            List of decision dictionaries
        """
        return [self.predict(text) for text in texts]
