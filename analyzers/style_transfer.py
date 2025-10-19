"""
Style Transfer - Suggests how to make text sound more like the user's style
"""

import re
import random
from typing import List, Dict, Optional
from collections import Counter

class StyleTransfer:
    def __init__(self):
        self.improvement_phrases = {
            'sentence_rhythm': [
                "Break up long sentences like you usually do",
                "Try your characteristic short, punchy sentences",
                "Vary sentence lengths like in your other writing"
            ],
            'vocabulary': [
                "Use more varied vocabulary like you typically do",
                "Incorporate some of your favorite descriptive words",
                "Try the specific terminology you often use"
            ],
            'emotional_tone': [
                "Add more of your characteristic emotional language",
                "Express opinions more directly like you usually do",
                "Use the conversational tone you're known for"
            ]
        }
    
    def suggest_improvements(self, text: str, user_dna: Dict, current_analysis: Dict) -> List[Dict]:
        """Generate suggestions to make text sound more like user's style"""
        suggestions = []
        
        # Sentence rhythm adjustments
        rhythm_suggestions = self._suggest_rhythm_improvements(text, user_dna)
        suggestions.extend(rhythm_suggestions)
        
        # Vocabulary matching
        vocab_suggestions = self._suggest_vocabulary_improvements(text, user_dna)
        suggestions.extend(vocab_suggestions)
        
        # Emotional tone alignment
        tone_suggestions = self._suggest_tone_improvements(text, user_dna)
        suggestions.extend(tone_suggestions)
        
        # Signature phrase incorporation
        phrase_suggestions = self._suggest_phrase_improvements(text, user_dna)
        suggestions.extend(phrase_suggestions)
        
        # Limit to most relevant suggestions
        return self._prioritize_suggestions(suggestions, current_analysis)[:5]
    
    def _suggest_rhythm_improvements(self, text: str, user_dna: Dict) -> List[Dict]:
        """Suggest improvements to match user's sentence rhythm"""
        suggestions = []
        
        # Analyze current text rhythm
        current_rhythm = self._analyze_sentence_rhythm(text)
        user_rhythm = user_dna['sentence_rhythm']
        
        # Check sentence length matching
        length_diff = abs(current_rhythm['avg_length'] - user_rhythm['avg_length'])
        if length_diff > 5:
            if current_rhythm['avg_length'] > user_rhythm['avg_length']:
                suggestions.append({
                    "type": "sentence_rhythm",
                    "message": "Break up long sentences like you usually do",
                    "priority": "high",
                    "details": f"Your average: {user_rhythm['avg_length']:.1f} words, Current: {current_rhythm['avg_length']:.1f} words"
                })
            else:
                suggestions.append({
                    "type": "sentence_rhythm", 
                    "message": "Try combining some short sentences for better flow",
                    "priority": "medium",
                    "details": f"Your average: {user_rhythm['avg_length']:.1f} words, Current: {current_rhythm['avg_length']:.1f} words"
                })
        
        # Check sentence variety
        current_variance = current_rhythm['length_variance']
        user_variance = user_rhythm['length_variance']
        
        if abs(current_variance - user_variance) > 10:
            if current_variance < user_variance:
                suggestions.append({
                    "type": "sentence_variety",
                    "message": "Vary sentence lengths more like you typically do",
                    "priority": "medium",
                    "details": "Mix short and long sentences for better rhythm"
                })
        
        return suggestions
    
    def _suggest_vocabulary_improvements(self, text: str, user_dna: Dict) -> List[Dict]:
        """Suggest vocabulary improvements to match user's style"""
        suggestions = []
        
        # Analyze current vocabulary
        current_words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        current_richness = len(set(current_words)) / len(current_words) if current_words else 0
        user_richness = user_dna['vocabulary_richness']
        
        # Vocabulary richness comparison
        if current_richness < user_richness * 0.7:
            suggestions.append({
                "type": "vocabulary",
                "message": "Use more varied vocabulary like in your other writing",
                "priority": "medium",
                "details": f"Your richness: {user_richness:.1%}, Current: {current_richness:.1%}"
            })
        
        # Suggest specific word replacements based on user's common words
        user_common_words = list(user_dna.get('word_frequency', {}).keys())[:10]
        current_common_words = [word for word, count in Counter(current_words).most_common(10)]
        
        # Find user's characteristic words not in current text
        missing_characteristic_words = [
            word for word in user_common_words 
            if word not in current_common_words and word not in ['that', 'with', 'this', 'have']
        ]
        
        if missing_characteristic_words:
            sample_word = random.choice(missing_characteristic_words[:3])
            suggestions.append({
                "type": "characteristic_vocabulary",
                "message": f"Try using '{sample_word}' or other words you frequently use",
                "priority": "low",
                "details": "Incorporate your signature vocabulary"
            })
        
        return suggestions
    
    def _suggest_tone_improvements(self, text: str, user_dna: Dict) -> List[Dict]:
        """Suggest tone improvements to match user's emotional style"""
        suggestions = []
        
        # Analyze current tone
        current_tone = self._analyze_emotional_tone(text)
        user_tone = user_dna['emotional_tone']
        
        # Find user's dominant tone
        user_dominant_tone = max(user_tone.items(), key=lambda x: x[1])
        
        if user_dominant_tone[1] > 0.05:  # If user has a clear tone preference
            current_dominant_tone = max(current_tone.items(), key=lambda x: x[1])
            
            if user_dominant_tone[0] != current_dominant_tone[0]:
                suggestions.append({
                    "type": "emotional_tone",
                    "message": f"Add more {user_dominant_tone[0]} language like you normally do",
                    "priority": "medium",
                    "details": self._get_tone_examples(user_dominant_tone[0])
                })
        
        return suggestions
    
    def _suggest_phrase_improvements(self, text: str, user_dna: Dict) -> List[Dict]:
        """Suggest incorporating user's signature phrases"""
        suggestions = []
        
        signature_phrases = user_dna.get('signature_phrases', [])
        text_lower = text.lower()
        
        # Find signature phrases not in current text
        missing_phrases = [phrase for phrase in signature_phrases if phrase not in text_lower]
        
        if missing_phrases:
            sample_phrase = random.choice(missing_phrases[:3])
            suggestions.append({
                "type": "signature_phrases",
                "message": f"Incorporate phrases like '{sample_phrase}' that you often use",
                "priority": "low", 
                "details": "Use your characteristic expressions"
            })
        
        return suggestions
    
    def _analyze_sentence_rhythm(self, text: str) -> Dict:
        """Analyze sentence rhythm of current text"""
        sentences = re.split(r'[.!?]+', text)
        lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        
        if not lengths:
            return {'avg_length': 0, 'length_variance': 0}
        
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        return {
            'avg_length': avg_length,
            'length_variance': variance
        }
    
    def _analyze_emotional_tone(self, text: str) -> Dict[str, float]:
        """Simple emotional tone analysis"""
        emotional_words = {
            'positive': ['love', 'great', 'amazing', 'wonderful', 'excellent', 'happy', 'joy'],
            'negative': ['hate', 'terrible', 'awful', 'horrible', 'bad', 'sad', 'angry'],
            'analytical': ['because', 'therefore', 'however', 'thus', 'consequently'],
            'confident': ['certainly', 'definitely', 'undoubtedly', 'clearly', 'obviously']
        }
        
        words = text.lower().split()
        total_words = len(words)
        
        if total_words == 0:
            return {emotion: 0.0 for emotion in emotional_words.keys()}
        
        tone_scores = {}
        for emotion, word_list in emotional_words.items():
            count = sum(1 for word in words if word in word_list)
            tone_scores[emotion] = count / total_words
        
        return tone_scores
    
    def _get_tone_examples(self, tone: str) -> str:
        """Get examples for different tones"""
        examples = {
            "positive": "Add words like 'great', 'excellent', 'wonderful'",
            "analytical": "Use logical connectors like 'because', 'therefore', 'however'",
            "confident": "Include confident phrases like 'clearly', 'definitely', 'undoubtedly'",
            "cautious": "Add cautious language like 'perhaps', 'might', 'could'"
        }
        return examples.get(tone, "Adjust the emotional tone to match your style")
    
    def _prioritize_suggestions(self, suggestions: List[Dict], current_analysis: Dict) -> List[Dict]:
        """Prioritize suggestions based on impact and current issues"""
        priority_weights = {
            "high": 3,
            "medium": 2, 
            "low": 1
        }
        
        # Sort by priority and relevance to current issues
        current_issue_types = [issue['type'] for issue in current_analysis.get('issues', [])]
        
        def suggestion_score(suggestion):
            score = priority_weights.get(suggestion.get('priority', 'low'), 1)
            # Boost score if related to current issues
            if any(issue_type in suggestion.get('type', '') for issue_type in current_issue_types):
                score += 1
            return score
        
        return sorted(suggestions, key=suggestion_score, reverse=True)
