"""
Perplexity Analyzer - Measures text predictability and creativity
"""

import numpy as np
from typing import List, Dict
import re

class PerplexityAnalyzer:
    def __init__(self):
        self.common_word_penalty = 0.1
        
    def analyze(self, text: str) -> List[Dict]:
        """Analyze text using perplexity-like metrics"""
        issues = []
        
        creativity_score = self._calculate_creativity_score(text)
        
        if creativity_score < 30:
            issues.append({
                "type": "low_creativity",
                "reason": "Text is very predictable",
                "suggestion": "Add more creative or unexpected word choices",
                "severity": "medium",
                "score": creativity_score
            })
        elif creativity_score > 80:
            issues.append({
                "type": "high_creativity", 
                "reason": "Text is highly unpredictable",
                "suggestion": "Consider if this is too chaotic for your audience",
                "severity": "low",
                "score": creativity_score
            })
        
        return issues
    
    def _calculate_creativity_score(self, text: str) -> float:
        """Calculate a creativity score (0-100) based on word diversity and rarity"""
        words = self._extract_words(text)
        
        if len(words) < 10:
            return 50.0
        
        # Type-token ratio (vocabulary diversity)
        unique_words = len(set(words))
        total_words = len(words)
        ttr = unique_words / total_words
        
        # Uncommon word ratio
        common_words = {'the', 'and', 'for', 'was', 'were', 'this', 'that', 'with', 'have', 'has', 'are', 'but', 'not'}
        uncommon_count = sum(1 for w in words if w not in common_words)
        uncommon_ratio = uncommon_count / total_words
        
        # Sentence length variation
        sentences = [s for s in text.split('.') if s.strip()]
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences]
            length_variance = np.var(sentence_lengths) / 10  # Normalized
        else:
            length_variance = 0.5
        
        # Combined creativity score
        creativity_score = (ttr * 0.4 + uncommon_ratio * 0.4 + length_variance * 0.2) * 100
        
        return max(0, min(100, creativity_score))
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from text"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return words