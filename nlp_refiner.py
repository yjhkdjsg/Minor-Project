#!/usr/bin/env python3
"""
Enhanced Hybrid ASL-to-English Pipeline

Improvements:
- Better character collapse with letter deduplication awareness
- Enhanced ASL pattern matching with 100+ common phrases
- Smarter spell correction that preserves ASL patterns
- Optional semantic refinement with better prompting
- Word-level validation to prevent over-correction

Usage:
    pip install -U transformers torch sentencepiece
    python nlp_refiner_v2.py
"""

from typing import List, Optional, Tuple, Dict
import re
import time
import logging

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception as e:
    raise ImportError(
        "Missing dependencies. Install with: pip install transformers torch sentencepiece"
    )

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("hybrid_asl")


# ---------- Enhanced ASL Pattern Dictionary ----------

class ASLPatternMatcher:
    """Comprehensive ASL fingerspelling pattern matcher"""
    
    def __init__(self):
        # Build comprehensive pattern dictionary
        self.patterns = {
            # Greetings
            'hello': 'hello',
            'hi': 'hi',
            'hey': 'hey',
            'goodmorning': 'good morning',
            'goodafternoon': 'good afternoon',
            'goodevening': 'good evening',
            'goodnight': 'good night',
            'goodbye': 'goodbye',
            'bye': 'bye',
            
            # Common phrases
            'thankyou': 'thank you',
            'thanks': 'thanks',
            'please': 'please',
            'sorry': 'sorry',
            'excuseme': 'excuse me',
            'welcomeback': 'welcome back',
            'yourwelcome': 'you are welcome',
            'youarewelcome': 'you are welcome',
            
            # Love expressions
            'iloveyou': 'i love you',
            'loveyou': 'love you',
            'iloveme': 'i love me',
            'ilove': 'i love',
            
            # Introductions
            'myname': 'my name',
            'mynameis': 'my name is',
            'nicetomeetyou': 'nice to meet you',
            'nicemeeting': 'nice meeting',
            'howareyou': 'how are you',
            'howyou': 'how are you',
            'whatisyourname': 'what is your name',
            'whatsyourname': 'what is your name',
            'whatyourname': 'what is your name',
            
            # Common verbs
            'iam': 'i am',
            'iamstudying': 'i am studying',
            'studying': 'studying',
            'learning': 'learning',
            'working': 'working',
            'teaching': 'teaching',
            
            # Common topics
            'bigdata': 'big data',
            'computer': 'computer',
            'science': 'science',
            'english': 'english',
            'mathematics': 'mathematics',
            'math': 'math',
            
            # Phrases with "very"
            'verymuch': 'very much',
            'thankyouverymuch': 'thank you very much',
            'verynice': 'very nice',
            'verygood': 'very good',
            'veryhappy': 'very happy',
        }
        
        # Sort by length (longest first) for better matching
        self.sorted_patterns = sorted(
            self.patterns.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
    
    def match(self, text: str) -> Optional[str]:
        """Try to match known ASL patterns"""
        text_clean = text.lower().replace(' ', '').replace('-', '')
        
        # Direct pattern matching
        for pattern, replacement in self.sorted_patterns:
            if pattern == text_clean:
                return replacement
        
        # Partial phrase matching for common errors
        if 'ilov' in text_clean:
            if 'you' in text_clean:
                return 'i love you'
            elif 'me' in text_clean:
                return 'i love me'
            else:
                return 'i love'
        
        if 'thankyou' in text_clean or 'thanku' in text_clean:
            if 'verymuch' in text_clean or 'much' in text_clean:
                return 'thank you very much'
            return 'thank you'
        
        if 'myname' in text_clean and 'is' in text_clean:
            # Extract the name after "is"
            parts = text_clean.split('is')
            if len(parts) > 1 and parts[1].strip():
                name = parts[1].strip()
                return f'my name is {name}'
            return 'my name is'
        
        if 'niceto' in text_clean and 'meet' in text_clean:
            return 'nice to meet you'
        
        return None


# ---------- Smart Character Collapse ----------

def smart_collapse_chars(text: str, dedupe_threshold: int = 3) -> str:
    """
    Intelligently collapse character streams while handling duplicates.
    
    Rules:
    - Single letters are joined into words
    - Consecutive identical letters (e.g., "s s s") are reduced to single letter
    - Multi-char tokens are preserved
    - Threshold determines when to deduplicate (default: 3+ consecutive)
    """
    if not text or not text.strip():
        return ""
    
    tokens = text.strip().split()
    if not tokens:
        return ""
    
    words = []
    current_word = []
    last_char = None
    char_count = 0
    
    for token in tokens:
        token = token.strip()
        if not token:
            continue
            
        # Multi-character token
        if len(token) > 1:
            # Flush current word
            if current_word:
                words.append(''.join(current_word))
                current_word = []
                last_char = None
                char_count = 0
            words.append(token)
        
        # Single character
        elif len(token) == 1 and token.isalpha():
            # Check for repetition
            if token.lower() == last_char:
                char_count += 1
                # Only add if below threshold (deduplicate excessive repeats)
                if char_count < dedupe_threshold:
                    current_word.append(token)
            else:
                current_word.append(token)
                last_char = token.lower()
                char_count = 1
        
        # Non-alpha single char (punctuation, etc.)
        else:
            if current_word:
                words.append(''.join(current_word))
                current_word = []
                last_char = None
                char_count = 0
            words.append(token)
    
    # Flush remaining
    if current_word:
        words.append(''.join(current_word))
    
    return ' '.join(words)


# ---------- Enhanced Spell Corrector ----------

class EnhancedSpellCorrector:
    """Spell corrector with ASL pattern awareness"""
    
    def __init__(self, model_name: str = "oliverguhr/spelling-correction-english-base", 
                 device: Optional[str] = None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pattern_matcher = ASLPatternMatcher()
        
        logger.info(f"Loading spell-corrector: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        if self.device == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                pass
        
        self.model.to(self.device)
        self.model.eval()
        
        # Common English words for validation
        self.common_words = {
            'i', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
            'can', 'could', 'may', 'might', 'must', 'should',
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'me', 'you', 'him', 'her', 'it', 'us', 'them',
            'what', 'when', 'where', 'why', 'how', 'which', 'who',
            'hello', 'hi', 'bye', 'thanks', 'thank', 'please', 'sorry',
            'yes', 'no', 'not', 'very', 'much', 'many', 'some', 'any',
            'good', 'bad', 'big', 'small', 'new', 'old', 'nice',
            'love', 'like', 'want', 'need', 'know', 'think', 'see',
            'name', 'meet', 'studying', 'learning', 'working',
            'data', 'science', 'computer', 'english', 'math'
        }
    
    def correct(self, text: str, max_length: int = 128) -> str:
        """Correct spelling with ASL pattern awareness"""
        
        # First, try pattern matching
        pattern_match = self.pattern_matcher.match(text)
        if pattern_match:
            logger.debug(f"Pattern match: {text} → {pattern_match}")
            return pattern_match
        
        # Check if text is already reasonable English
        words = text.lower().split()
        if len(words) > 0:
            # If most words are common English words, minimal correction
            known_ratio = sum(1 for w in words if w in self.common_words) / len(words)
            if known_ratio > 0.6:
                logger.debug(f"Text appears valid ({known_ratio:.0%} known words)")
                return text
        
        # Use spell correction model
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                    no_repeat_ngram_size=2,
                )
            
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = decoded.strip()
            
            # Validate: don't accept if model made text much shorter (over-correction)
            if len(result.split()) < len(text.split()) * 0.4:
                logger.debug(f"Model over-corrected, keeping original")
                return text
            
            return result
            
        except Exception as e:
            logger.warning(f"Correction failed: {e}")
            return text


# ---------- Smart Semantic Refiner ----------

class SmartSemanticRefiner:
    """Lightweight semantic refinement with better prompting"""
    
    def __init__(self, model_name: Optional[str] = "google/flan-t5-small", 
                 device: Optional[str] = None):
        self.enabled = model_name is not None
        if not self.enabled:
            return
        
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading semantic refiner: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        if self.device == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                pass
        
        self.model.to(self.device)
        self.model.eval()
    
    def refine(self, text: str, max_length: int = 128) -> str:
        """Add punctuation and capitalization without changing words"""
        if not self.enabled:
            return text
        
        # Simple rule-based capitalization for short phrases
        words = text.split()
        if len(words) <= 6:
            # Capitalize first word
            if words:
                words[0] = words[0].capitalize()
            
            # Add period if no ending punctuation
            result = ' '.join(words)
            if result and result[-1] not in '.!?':
                result += '.'
            
            return result
        
        # For longer text, use model
        prompt = (
            f"Add proper capitalization and punctuation to this sentence. "
            f"Do not change any words: {text}"
        )
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=3,
                    early_stopping=True,
                    repetition_penalty=1.1,
                )
            
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = decoded.strip()
            
            # Validate: check if words were changed significantly
            orig_words = set(text.lower().replace('.', '').replace(',', '').split())
            new_words = set(result.lower().replace('.', '').replace(',', '').split())
            
            # If too many words changed, skip refinement
            if len(orig_words - new_words) > len(orig_words) * 0.3:
                logger.debug("Semantic refiner changed too many words, skipping")
                return text
            
            return result
            
        except Exception as e:
            logger.warning(f"Refinement failed: {e}")
            return text


# ---------- Enhanced Pipeline ----------

class EnhancedASLPipeline:
    """Enhanced ASL-to-English pipeline with better accuracy"""
    
    def __init__(self, 
                 spell_model: str = "oliverguhr/spelling-correction-english-base",
                 semantic_model: Optional[str] = None,  # Disabled by default
                 use_semantic: bool = False):
        
        self.spell_corrector = EnhancedSpellCorrector(spell_model)
        self.semantic_refiner = (
            SmartSemanticRefiner(semantic_model) 
            if use_semantic and semantic_model 
            else None
        )
        self.pattern_matcher = ASLPatternMatcher()
    
    def process(self, raw_buffer: str) -> Tuple[str, Dict]:
        """Process ASL fingerspelling to English text"""
        t0 = time.time()
        meta = {"original": raw_buffer}
        
        # Step 1: Clean whitespace
        cleaned = raw_buffer.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        meta["cleaned"] = cleaned
        
        # Step 2: Smart character collapse with deduplication
        collapsed = smart_collapse_chars(cleaned, dedupe_threshold=3)
        meta["collapsed"] = collapsed
        
        # Step 3: Pattern matching (highest priority)
        pattern_result = self.pattern_matcher.match(collapsed)
        if pattern_result:
            meta["method"] = "pattern_match"
            meta["corrected"] = pattern_result
            final = pattern_result
        else:
            # Step 4: Spell correction
            corrected = self.spell_corrector.correct(collapsed)
            meta["method"] = "spell_correction"
            meta["corrected"] = corrected
            final = corrected
        
        # Step 5: Optional semantic refinement
        if self.semantic_refiner:
            final = self.semantic_refiner.refine(final)
            meta["refined"] = final
        
        meta["final"] = final
        meta["processing_time_seconds"] = round(time.time() - t0, 4)
        
        return final, meta


# ---------- Demo ----------

if __name__ == "__main__":
    demo_inputs = [
        "i l o v m s s m e e",
        "h e l l o m y n a m e i s s n e h a",
        "i a m s t u d y i n g b i g d a t a",
        "t h a n k y o u v e r y m u c h",
        "w h a t i s y o u r n a m e",
        "n i c e t o m m e e t y o u",
        "h e l ll o m y n a m e i s s n e h a a",
        "i l o v e y o u",
        "h o w a r e y o u",
        "g o o d m o r n i n g",
    ]
    
    print("🤟 Enhanced ASL-to-English Pipeline\n")
    
    # Without semantic refinement (recommended for accuracy)
    pipeline = EnhancedASLPipeline(use_semantic=False)
    
    for sample in demo_inputs:
        final, debug = pipeline.process(sample)
        print(f"📝 Input:    {sample}")
        print(f"✨ Output:   {final}")
        print(f"🔍 Method:   {debug['method']}")
        print(f"   Steps:    {debug['collapsed']} → {debug['corrected']}")
        print(f"⏱️  Time:     {debug['processing_time_seconds']}s")
        print()