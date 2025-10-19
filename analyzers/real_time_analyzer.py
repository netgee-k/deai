"""
Real-time Analyzer - Fast analysis for live typing
"""

from typing import Dict, List
import re

class RealTimeAnalyzer:
    def __init__(self):
        self.quick_patterns = [
            (r"\b(utilize|commence|terminate)\b", "formal_word"),
            (r"\b(in order to|due to the fact that)\b", "redundant_phrase"),
            (r"\b(delve into|tapestry|realm)\b", "ai_cliche"),
        ]
    
    def quick_analyze(self, text: str) -> Dict:
        """Quick analysis suitable for real-time typing"""
        if len(text.strip()) < 20:
            return {"score": 50, "issues": []}
        
        issues = []
        
        # Quick formal word check
        formal_count = len(re.findall(r"\b(utilize|commence|terminate|assistance)\b", text, re.IGNORECASE))
        if formal_count > 2:
            issues.append({
                "type": "too_formal",
                "message": "Consider using simpler words"
            })
        
        # Quick sentence length check
        sentences = [s for s in text.split('.') if s.strip()]
        if len(sentences) > 3:
            lengths = [len(s.split()) for s in sentences]
            if max(lengths) - min(lengths) < 3:
                issues.append({
                    "type": "uniform_rhythm", 
                    "message": "Vary your sentence lengths"
                })
        
        score = max(0, 100 - (len(issues) * 10))
        
        return {
            "score": score,
            "issues": issues,
            "ready": len(text) > 100  # Only show detailed analysis for substantial text
        }