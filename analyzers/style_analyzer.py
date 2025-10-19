"""
Style Analyzer - Learns and compares against your writing style
"""

import re
import numpy as np
from collections import Counter
from typing import List, Dict, Optional
import json
import os

class StyleAnalyzer:
    def __init__(self):
        self.user_profiles = {}
        self.load_profiles()
    
    def train_on_documents(self, documents: List[str], user_id: str):
        """Train on user's documents to learn their style"""
        corpus = " ".join(documents)
        
        profile = {
            'sentence_lengths': self._analyze_sentence_lengths(corpus),
            'word_frequency': self._analyze_word_frequency(corpus),
            'contraction_ratio': self._analyze_contractions(corpus),
        }
        
        self.user_profiles[user_id] = profile
        self.save_profiles()
        
        return profile
    
    def analyze_against_profile(self, text: str, user_id: str) -> List[Dict]:
        """Analyze text against user's style profile"""
        if user_id not in self.user_profiles:
            return [{
                "type": "no_profile",
                "reason": "No style profile trained yet",
                "suggestion": "Upload your writing to train your style model",
                "severity": "low"
            }]
        
        profile = self.user_profiles[user_id]
        issues = []
        
        # Check sentence length patterns
        current_lengths = self._analyze_sentence_lengths(text)
        if current_lengths and profile['sentence_lengths']:
            current_avg = np.mean(current_lengths)
            profile_avg = np.mean(profile['sentence_lengths'])
            
            if abs(current_avg - profile_avg) > 5:
                issues.append({
                    "type": "sentence_rhythm",
                    "reason": f"Sentence length differs from your usual style",
                    "suggestion": f"Your average: {profile_avg:.1f} words, current: {current_avg:.1f}",
                    "severity": "medium"
                })
        
        return issues
    
    def calculate_style_match(self, text: str, user_id: str) -> float:
        """Calculate percentage match with user's style (0-100)"""
        if user_id not in self.user_profiles:
            return 50.0  # Default score
        
        profile = self.user_profiles[user_id]
        scores = []
        
        # Sentence length similarity
        current_lengths = self._analyze_sentence_lengths(text)
        if current_lengths and profile['sentence_lengths']:
            current_avg = np.mean(current_lengths)
            profile_avg = np.mean(profile['sentence_lengths'])
            length_similarity = 1.0 - min(1.0, abs(current_avg - profile_avg) / 20.0)
            scores.append(length_similarity)
        
        return np.mean(scores) * 100 if scores else 50.0
    
    def _analyze_sentence_lengths(self, text: str) -> List[float]:
        sentences = re.split(r'[.!?]+', text)
        return [len(s.split()) for s in sentences if s.strip()]
    
    def _analyze_word_frequency(self, corpus: str) -> Dict[str, float]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', corpus.lower())
        common_words = set(['the', 'and', 'for', 'was', 'were', 'this', 'that'])
        filtered_words = [w for w in words if w not in common_words]
        
        word_counts = Counter(filtered_words)
        total = sum(word_counts.values())
        
        return {word: count/total for word, count in word_counts.most_common(100)}
    
    def _analyze_contractions(self, text: str) -> float:
        contractions = len(re.findall(r"\w+'\w+", text))
        total_words = len(text.split())
        return contractions / total_words if total_words > 0 else 0
    
    def _check_vocabulary_match(self, text: str, profile_frequency: Dict) -> float:
        text_words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        common_words = set(['the', 'and', 'for', 'was', 'were', 'this', 'that'])
        filtered_words = [w for w in text_words if w not in common_words]
        
        if not filtered_words:
            return 0.0
        
        matches = sum(1 for word in filtered_words if word in profile_frequency)
        return matches / len(filtered_words)
    
    def save_profiles(self):
        """Save user profiles to disk"""
        with open('data/user_profiles.json', 'w') as f:
            json.dump(self.user_profiles, f)
    
    def load_profiles(self):
        """Load user profiles from disk"""
        try:
            with open('data/user_profiles.json', 'r') as f:
                self.user_profiles = json.load(f)
        except FileNotFoundError:
            self.user_profiles = {}