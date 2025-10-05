"""
Ultra-fast punctuation restoration for streaming STT
Combines rule-based heuristics with ML-based punctuation restoration
"""

import re
import spacy
from typing import List, Optional
import time

# Try to import deepmultilingualpunctuation for better results
try:
    from deepmultilingualpunctuation import PunctuationModel
    DEEP_PUNCTUATION_AVAILABLE = True
except ImportError:
    DEEP_PUNCTUATION_AVAILABLE = False


class FastPunctuationRestorer:
    """Ultra-fast punctuation restoration optimized for streaming STT"""

    def __init__(self, use_deep_model: bool = True):
        # Initialize deep punctuation model if available and requested
        self.deep_model = None
        if use_deep_model and DEEP_PUNCTUATION_AVAILABLE:
            try:
                self.deep_model = PunctuationModel()
                print("Loaded deep multilingual punctuation model")
            except Exception as e:
                print(f"Failed to load deep model: {e}, falling back to rule-based")

        # Load minimal spaCy model for sentence boundaries (fallback)
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            if not self.nlp.has_pipe("sentencizer"):
                self.nlp.add_pipe("sentencizer")
        except OSError:
            # Fallback to basic English with sentencizer
            from spacy.lang.en import English
            self.nlp = English()
            self.nlp.add_pipe("sentencizer")

        # Core wh-question words (safe to mark as question when first token)
        self.wh_words = {
            "what", "where", "when", "why", "who", "whom", "which", "whose", "how"
        }
        # Invertible auxiliaries that can form questions only if they start the sentence
        self.aux_first_only = {
            "can", "could", "would", "should", "will", "do", "does", "did",
            "is", "are", "am", "was", "were", "have", "has", "had"
        }

        # Pause indicators for comma placement
        self.pause_indicators = {
            "and", "but", "or", "so", "yet", "however", "therefore", "meanwhile",
            "first", "second", "third", "finally", "also", "additionally"
        }

    def restore_punctuation(self, text: str, confidence_threshold: float = 0.7) -> str:
        """
        Restore punctuation to text with best available method

        Args:
            text: Raw text without punctuation
            confidence_threshold: Minimum confidence for punctuation placement

        Returns:
            Text with restored punctuation
        """
        if not text.strip():
            return text

        # Try deep model first if available (prefer accuracy over micro-latency on final text)
        if self.deep_model:
            try:
                start_time = time.perf_counter()
                result = self.deep_model.restore_punctuation(text)
                processing_time = (time.perf_counter() - start_time) * 1000
                # Prefer deep model result even if slower; finalization can tolerate ~150ms
                if processing_time > 150:
                    print(f"Deep model slow ({processing_time:.1f}ms), but using for accuracy")
                return result
            except Exception as e:
                print(f"Deep model failed: {e}, using fallback")

        # Fall back to rule-based approach
        return self._rule_based_punctuation(text)

    def _rule_based_punctuation(self, text: str) -> str:
        """Rule-based punctuation restoration (original fast method)"""
        # Clean and normalize text
        text = self._normalize_text(text)

        # Split into potential sentences
        sentences = self._split_sentences(text)

        # Process each sentence
        punctuated_sentences = []
        for sentence in sentences:
            if sentence.strip():
                punctuated = self._add_sentence_punctuation(sentence.strip())
                punctuated = self._add_internal_punctuation(punctuated)
                punctuated_sentences.append(punctuated)

        return " ".join(punctuated_sentences)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        # Remove existing punctuation and extra spaces
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into potential sentences using fast heuristics"""
        # Use spaCy for basic sentence segmentation
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]

        # Fallback: split on long pauses or obvious boundaries
        if len(sentences) == 1 and len(text.split()) > 15:
            # Split long sentences on conjunctions
            words = text.split()
            sentences = []
            current = []

            for i, word in enumerate(words):
                current.append(word)
                # Split on certain conjunctions if sentence is getting long
                if (len(current) > 8 and word.lower() in ["and", "but", "so"] and
                    i < len(words) - 3):
                    sentences.append(" ".join(current[:-1]))
                    current = [word]

            if current:
                sentences.append(" ".join(current))

        return sentences

    def _add_sentence_punctuation(self, sentence: str) -> str:
        """Add end-of-sentence punctuation"""
        words = sentence.lower().split()
        if not words:
            return sentence

        first_word = words[0]

        # Question detection: wh-words always, auxiliaries only if first
        if first_word in self.wh_words or first_word in self.aux_first_only:
            return sentence + "?"

        # Default to period
        return sentence + "."

    def _add_internal_punctuation(self, sentence: str) -> str:
        """Add commas and other internal punctuation"""
        words = sentence.split()
        if len(words) < 4:
            return sentence

        result = []
        for i, word in enumerate(words):
            result.append(word)

            # Add comma after pause indicators (but not at end)
            if (i < len(words) - 2 and
                word.lower().rstrip('.,!?') in self.pause_indicators):
                if not word.endswith(('.', '!', '?', ',')):
                    result[-1] = word + ","

        return " ".join(result)


# Integration example
class StreamingSTTWithPunctuation:
    """Wrapper to add fast punctuation to streaming STT"""

    def __init__(self, stt_service):
        self.stt_service = stt_service
        self.punctuation_restorer = FastPunctuationRestorer()
        self._buffer = []
        self._last_punctuation_time = 0

    def process_interim_text(self, text: str) -> str:
        """Process interim transcription (no punctuation yet)"""
        return text

    def process_final_text(self, text: str) -> str:
        """Process final transcription with punctuation restoration"""
        start_time = time.perf_counter()

        # Restore punctuation
        punctuated_text = self.punctuation_restorer.restore_punctuation(text)

        processing_time = (time.perf_counter() - start_time) * 1000
        print(f"Punctuation restoration: {processing_time:.1f}ms")

        return punctuated_text


if __name__ == "__main__":
    # Test the punctuation restorer
    restorer = FastPunctuationRestorer()

    test_texts = [
        "can you hear me",
        "hello how are you today",
        "what time is it and where should we meet",
        "i think we should go but i am not sure",
        "first we need to get the data then we can analyze it"
    ]

    for text in test_texts:
        start = time.perf_counter()
        result = restorer.restore_punctuation(text)
        latency = (time.perf_counter() - start) * 1000
        print(f"{text} -> {result} ({latency:.1f}ms)")
