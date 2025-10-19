"""
Writing DNA Analyzer - Deep style analysis and fingerprinting
"""

import re
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import math

class WritingDNA:
    def __init__(self):
        self.fingerprints = {}
    
    def create_fingerprint(self, documents: List[str], user_id: str) -> Dict:
        """Create a comprehensive writing fingerprint from documents"""
        corpus = " ".join(documents)
        
        fingerprint = {
            'vocabulary_richness': self._calculate_vocabulary_richness(corpus),
            'sentence_rhythm': self._analyze_sentence_rhythm(corpus),
            'emotional_tone': self._analyze_emotional_tone(corpus),
            'signature_phrases': self._find_signature_phrases(corpus),
            'readability_level': self._calculate_readability(corpus),
            'writing_patterns': self._analyze_writing_patterns(corpus),
            'complexity_metrics': self._calculate_complexity_metrics(corpus)
        }
        
        self.fingerprints[user_id] = fingerprint
        return fingerprint
    
    def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calculate vocabulary richness (Type-Token Ratio)"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        return unique_words / len(words)
    
    def _analyze_sentence_rhythm(self, text: str) -> Dict:
        """Analyze sentence structure and rhythm patterns"""
        sentences = self._split_sentences(text)
        lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        
        if not lengths:
            return {
                'avg_length': 0,
                'length_variance': 0,
                'short_sentence_ratio': 0,
                'long_sentence_ratio': 0,
                'rhythm_consistency': 0
            }
        
        avg_length = np.mean(lengths)
        length_variance = np.var(lengths)
        
        # Calculate ratios
        short_ratio = len([l for l in lengths if l < 8]) / len(lengths)
        long_ratio = len([l for l in lengths if l > 20]) / len(lengths)
        
        # Rhythm consistency (how predictable sentence lengths are)
        rhythm_consistency = 1.0 - (length_variance / (avg_length ** 2)) if avg_length > 0 else 0
        
        return {
            'avg_length': avg_length,
            'length_variance': length_variance,
            'short_sentence_ratio': short_ratio,
            'long_sentence_ratio': long_ratio,
            'rhythm_consistency': min(1.0, rhythm_consistency)
        }
    
    def _analyze_emotional_tone(self, text: str) -> Dict[str, float]:
        """Analyze emotional tone using word lists"""
        emotional_words = {
            'positive': [
                'love', 'great', 'amazing', 'wonderful', 'excellent', 'beautiful',
                'fantastic', 'brilliant', 'awesome', 'perfect', 'happy', 'joy',
                'pleasure', 'delight', 'marvelous', 'splendid', 'outstanding'
            ],
            'negative': [
                'hate', 'terrible', 'awful', 'horrible', 'bad', 'worst',
                'disappointing', 'unfortunate', 'tragic', 'miserable', 'sad',
                'angry', 'frustrating', 'annoying', 'disgusting', 'painful'
            ],
            'analytical': [
                'because', 'therefore', 'however', 'thus', 'consequently',
                'furthermore', 'moreover', 'nevertheless', 'accordingly',
                'hence', 'whereas', 'although', 'despite', 'notwithstanding'
            ],
            'confident': [
                'certainly', 'definitely', 'undoubtedly', 'clearly', 'obviously',
                'absolutely', 'indeed', 'unquestionably', 'assuredly', 'decidedly'
            ],
            'cautious': [
                'perhaps', 'maybe', 'possibly', 'potentially', 'might',
                'could', 'seems', 'appears', 'suggests', 'indicates'
            ]
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
    
    def _find_signature_phrases(self, text: str, min_occurrences: int = 2) -> List[str]:
        """Find frequently used phrases that could be signature expressions"""
        sentences = self._split_sentences(text)
        all_phrases = []
        
        for sentence in sentences:
            words = [w.lower() for w in sentence.strip().split() if len(w) > 2]
            
            # Extract 2-4 word phrases
            for phrase_length in range(2, 5):
                for i in range(len(words) - phrase_length + 1):
                    phrase = ' '.join(words[i:i + phrase_length])
                    if len(phrase) > 8:  # Meaningful phrases only
                        all_phrases.append(phrase)
        
        # Count and filter phrases
        phrase_counts = Counter(all_phrases)
        signature_phrases = [
            phrase for phrase, count in phrase_counts.most_common(20)
            if count >= min_occurrences and self._is_meaningful_phrase(phrase)
        ]
        
        return signature_phrases[:10]  # Return top 10
    
    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if a phrase is meaningful (not just common word combinations)"""
        common_phrases = {
            'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'been',
            'are', 'was', 'were', 'will', 'would', 'should', 'could', 'about'
        }
        
        words = phrase.split()
        if len(words) < 2:
            return False
        
        # Filter out phrases that are too common
        if all(word in common_phrases for word in words):
            return False
        
        return True
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        sentences = self._split_sentences(text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        total_sentences = len(sentences)
        total_words = len(words)
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease formula
        try:
            score = 206.835 - (1.015 * (total_words / total_sentences)) - (84.6 * (total_syllables / total_words))
            return max(0, min(100, score))
        except ZeroDivisionError:
            return 0.0
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximate)"""
        word = word.lower().strip()
        if not word:
            return 0
        
        # Simple syllable counting
        vowels = "aeiouy"
        count = 0
        prev_char_vowel = False
        
        for char in word:
            if char in vowels and not prev_char_vowel:
                count += 1
                prev_char_vowel = True
            else:
                prev_char_vowel = False
        
        # Adjust for common exceptions
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2:
            count += 1
        if count == 0:
            count = 1
            
        return count
    
    def _analyze_writing_patterns(self, text: str) -> Dict:
        """Analyze specific writing patterns and habits"""
        sentences = self._split_sentences(text)
        
        # Contraction usage
        contractions = len(re.findall(r"\w+'\w+", text))
        total_words = len(text.split())
        contraction_ratio = contractions / total_words if total_words > 0 else 0
        
        # Transition words usage
        transition_words = [
            'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
            'consequently', 'additionally', 'similarly', 'likewise', 'otherwise'
        ]
        transition_count = sum(1 for word in text.lower().split() if word in transition_words)
        transition_ratio = transition_count / total_words if total_words > 0 else 0
        
        # Question usage
        questions = len([s for s in sentences if s.strip().endswith('?')])
        question_ratio = questions / len(sentences) if sentences else 0
        
        # Exclamation usage
        exclamations = len([s for s in sentences if s.strip().endswith('!')])
        exclamation_ratio = exclamations / len(sentences) if sentences else 0
        
        return {
            'contraction_ratio': contraction_ratio,
            'transition_ratio': transition_ratio,
            'question_ratio': question_ratio,
            'exclamation_ratio': exclamation_ratio,
            'formality_score': 1.0 - contraction_ratio  # Higher = more formal
        }
    
    def _calculate_complexity_metrics(self, text: str) -> Dict:
        """Calculate various complexity metrics"""
        words = text.split()
        sentences = self._split_sentences(text)
        
        if not words or not sentences:
            return {
                'avg_word_length': 0,
                'sentence_complexity': 0,
                'lexical_diversity': 0,
                'descriptive_density': 0
            }
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Sentence complexity (clauses per sentence)
        clause_indicators = ['that', 'which', 'who', 'when', 'where', 'because', 'although']
        clauses = sum(1 for word in text.lower().split() if word in clause_indicators)
        sentence_complexity = clauses / len(sentences)
        
        # Lexical diversity (unique words ratio)
        lexical_diversity = len(set(words)) / len(words)
        
        # Descriptive density (adjectives and adverbs)
        descriptive_words = len([word for word in words if word.endswith(('ly', 'ful', 'ous', 'ive'))])
        descriptive_density = descriptive_words / len(words)
        
        return {
            'avg_word_length': avg_word_length,
            'sentence_complexity': sentence_complexity,
            'lexical_diversity': lexical_diversity,
            'descriptive_density': descriptive_density
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def compare_with_fingerprint(self, text: str, user_id: str) -> Dict[str, float]:
        """Compare new text with existing fingerprint"""
        if user_id not in self.fingerprints:
            return {"error": "No fingerprint found for user"}
        
        fingerprint = self.fingerprints[user_id]
        current_analysis = self.create_fingerprint([text], "temp")
        
        similarity_scores = {}
        
        # Compare various metrics
        similarity_scores['vocabulary'] = 1.0 - abs(
            fingerprint['vocabulary_richness'] - current_analysis['vocabulary_richness']
        )
        
        similarity_scores['sentence_rhythm'] = 1.0 - abs(
            fingerprint['sentence_rhythm']['avg_length'] - current_analysis['sentence_rhythm']['avg_length']
        ) / 20  # Normalize
        
        similarity_scores['readability'] = 1.0 - abs(
            fingerprint['readability_level'] - current_analysis['readability_level']
        ) / 100  # Normalize
        
        # Signature phrase match
        current_phrases = set(current_analysis['signature_phrases'])
        original_phrases = set(fingerprint['signature_phrases'])
        phrase_overlap = len(current_phrases.intersection(original_phrases))
        similarity_scores['signature_phrases'] = phrase_overlap / max(len(original_phrases), 1)
        
        # Overall similarity (weighted average)
        weights = {
            'vocabulary': 0.3,
            'sentence_rhythm': 0.3,
            'readability': 0.2,
            'signature_phrases': 0.2
        }
        
        overall_similarity = sum(
            similarity_scores[metric] * weight 
            for metric, weight in weights.items()
        )
        
        similarity_scores['overall'] = overall_similarity
        
        return similarity_scores
    
    def get_writing_personality(self, user_id: str) -> Dict:
        """Generate a writing personality profile"""
        if user_id not in self.fingerprints:
            return {"error": "No fingerprint found"}
        
        fingerprint = self.fingerprints[user_id]
        
        personality = {
            'traits': [],
            'style_description': '',
            'strengths': [],
            'suggestions': []
        }
        
        # Analyze traits based on metrics
        if fingerprint['vocabulary_richness'] > 0.4:
            personality['traits'].append('Eloquent')
            personality['strengths'].append('Rich vocabulary')
        elif fingerprint['vocabulary_richness'] < 0.2:
            personality['traits'].append('Direct')
            personality['strengths'].append('Clear communication')
        
        rhythm = fingerprint['sentence_rhythm']
        if rhythm['length_variance'] > 20:
            personality['traits'].append('Dynamic')
            personality['strengths'].append('Engaging rhythm')
        elif rhythm['length_variance'] < 5:
            personality['traits'].append('Consistent')
            personality['strengths'].append('Reliable pacing')
        
        # Emotional tone
        dominant_tone = max(fingerprint['emotional_tone'].items(), key=lambda x: x[1])
        if dominant_tone[1] > 0.05:
            personality['traits'].append(dominant_tone[0].title())
        
        # Formality
        formality = fingerprint['writing_patterns']['formality_score']
        if formality > 0.7:
            personality['traits'].append('Formal')
        elif formality < 0.3:
            personality['traits'].append('Conversational')
        
        # Generate style description
        if personality['traits']:
            primary_traits = personality['traits'][:2]
            personality['style_description'] = f"A {', '.join(primary_traits).lower()} writing style"
        
        # Suggestions for improvement
        if fingerprint['vocabulary_richness'] < 0.25:
            personality['suggestions'].append('Try incorporating more varied vocabulary')
        if rhythm['length_variance'] < 8:
            personality['suggestions'].append('Vary sentence lengths for better rhythm')
        if fingerprint['readability_level'] < 30:
            personality['suggestions'].append('Consider simplifying complex sentences')
        
        return personality
