"""
AI Smell Detector - Flags common AI writing patterns
"""

import re
from typing import List, Dict

class AISmellDetector:
    def __init__(self):
        self.formal_words = {
            "utilize": "use", "commence": "begin/start", "terminate": "end",
            "assistance": "help", "approximately": "about", "purchase": "buy",
            "inquire": "ask", "demonstrate": "show", "facilitate": "help",
            "additional": "more", "implement": "do", "leverage": "use"
        }
        
        self.redundant_phrases = {
            "in order to": "to", "due to the fact that": "because",
            "at this point in time": "now", "each and every": "each",
            "basic fundamentals": "fundamentals", "future plans": "plans",
            "past history": "history", "unexpected surprise": "surprise"
        }
        
        self.ai_cliches = [
            "delve into", "tapestry", "realm", "landscape", "journey",
            "testament to", "in the world of", "it is important to",
            "it is worth noting", "as we navigate", "unlock the potential"
        ]

    def detect(self, text: str) -> List[Dict]:
        issues = []
        
        # Check formal words
        for formal, simple in self.formal_words.items():
            for match in re.finditer(rf'\b{formal}\b', text, re.IGNORECASE):
                issues.append({
                    "type": "formal_word",
                    "text": match.group(),
                    "reason": f"Overly formal word",
                    "suggestion": f"Try '{simple}'",
                    "severity": "low",
                    "position": match.start()
                })
        
        # Check redundant phrases
        for phrase, replacement in self.redundant_phrases.items():
            for match in re.finditer(re.escape(phrase), text, re.IGNORECASE):
                issues.append({
                    "type": "redundant_phrase",
                    "text": match.group(),
                    "reason": "Wordy phrase",
                    "suggestion": f"Use '{replacement}'",
                    "severity": "medium",
                    "position": match.start()
                })
        
        # Check AI cliches
        for cliche in self.ai_cliches:
            for match in re.finditer(re.escape(cliche), text, re.IGNORECASE):
                issues.append({
                    "type": "ai_cliche",
                    "text": match.group(),
                    "reason": "Common AI writing pattern",
                    "suggestion": "Use more original phrasing",
                    "severity": "medium",
                    "position": match.start()
                })
        
        # Passive voice detection
        passive_issues = self._detect_passive_voice(text)
        issues.extend(passive_issues)
        
        return issues
    
    def _detect_passive_voice(self, text: str) -> List[Dict]:
        issues = []
        passive_patterns = [
            r"\b(was|were) \w+ed\b",
            r"\b(has|have) been \w+ed\b",
            r"\b(had) been \w+ed\b"
        ]
        
        for pattern in passive_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                issues.append({
                    "type": "passive_voice",
                    "text": match.group(),
                    "reason": "Passive voice can weaken writing",
                    "suggestion": "Consider active voice",
                    "severity": "low",
                    "position": match.start()
                })
        
        return issues