"""
Real AI Analyzer - Uses transformer models for actual perplexity calculation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
import re

class RealAIAnalyzer:
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialized = False
    
    def _initialize_model(self):
        """Lazy initialization of the model"""
        if self._initialized:
            return
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()  # Set to evaluation mode
            self._initialized = True
            print(f"✅ Real AI analyzer initialized with {self.model_name}")
        except Exception as e:
            print(f"❌ Failed to initialize AI model: {e}")
            # Fallback to basic analysis
            self._initialized = False
    
    def calculate_real_perplexity(self, text: str) -> float:
        """Calculate actual perplexity using transformer model"""
        if not text.strip():
            return 100.0  # Neutral score for empty text
            
        self._initialize_model()
        
        if not self._initialized:
            # Fallback to simple approximation
            return self._approximate_perplexity(text)
        
        try:
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
        except Exception as e:
            print(f"Perplexity calculation failed: {e}")
            return self._approximate_perplexity(text)
    
    def _approximate_perplexity(self, text: str) -> float:
        """Fallback perplexity approximation when model isn't available"""
        words = text.split()
        if len(words) < 5:
            return 50.0
        
        # Simple heuristic based on sentence structure and vocabulary
        score = 50.0
        
        # More unique words = higher perplexity
        unique_ratio = len(set(words)) / len(words)
        score += unique_ratio * 30
        
        # Longer sentences = potentially more predictable
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        score -= min(avg_sentence_length / 10, 20)
        
        return max(10, min(100, score))
    
    def analyze_with_ai(self, text: str, perplexity: float = None) -> Dict:
        """Analyze text using real AI model insights"""
        if perplexity is None:
            perplexity = self.calculate_real_perplexity(text)
        
        issues = []
        
        # Interpret perplexity scores
        if perplexity < 20:
            issues.append({
                "type": "low_perplexity",
                "text": f"Perplexity: {perplexity:.1f}",
                "reason": "Text is highly predictable (typical of AI-generated content)",
                "suggestion": "Add more creative or unexpected word choices",
                "severity": "high"
            })
        elif perplexity < 40:
            issues.append({
                "type": "low_perplexity", 
                "text": f"Perplexity: {perplexity:.1f}",
                "reason": "Text shows some predictability patterns",
                "suggestion": "Consider varying sentence structure and vocabulary",
                "severity": "medium"
            })
        elif perplexity > 80:
            issues.append({
                "type": "high_perplexity",
                "text": f"Perplexity: {perplexity:.1f}",
                "reason": "Text shows high creativity and unpredictability",
                "suggestion": "Great! This sounds very human-like",
                "severity": "low"
            })
        
        return {
            "perplexity": perplexity,
            "ai_likelihood": self.get_ai_likelihood(perplexity),
            "issues": issues
        }
    
    def get_ai_likelihood(self, perplexity: float) -> str:
        """Convert perplexity to AI likelihood assessment"""
        if perplexity < 20:
            return "Very High"
        elif perplexity < 35:
            return "High"
        elif perplexity < 50:
            return "Moderate"
        elif perplexity < 70:
            return "Low"
        else:
            return "Very Low"
    
    def compare_with_training_data(self, text: str, training_texts: List[str]) -> Dict:
        """Compare text with training data distribution"""
        if not training_texts:
            return {"error": "No training data provided"}
        
        # Calculate average perplexity of training data
        training_perplexities = []
        for train_text in training_texts[:5]:  # Sample first 5 for efficiency
            try:
                train_ppl = self.calculate_real_perplexity(train_text)
                training_perplexities.append(train_ppl)
            except:
                continue
        
        if not training_perplexities:
            return {"error": "Could not analyze training data"}
        
        avg_training_ppl = sum(training_perplexities) / len(training_perplexities)
        current_ppl = self.calculate_real_perplexity(text)
        
        similarity = 1.0 - abs(current_ppl - avg_training_ppl) / max(avg_training_ppl, current_ppl)
        
        return {
            "current_perplexity": current_ppl,
            "average_training_perplexity": avg_training_ppl,
            "similarity_score": similarity,
            "interpretation": "Very similar" if similarity > 0.8 else "Similar" if similarity > 0.6 else "Different"
        }
