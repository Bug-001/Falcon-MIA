import re
import math
import random
import string
from typing import Dict, Any, List, Set, Optional, Tuple, Union
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import spacy
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer

_initiated = False
if not _initiated:
    try:
        g_nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Downloading English language model...")
        from subprocess import check_call
        check_call(["python", "-m", "spacy", "download", 'en_core_web_sm'])
        g_nlp = spacy.load('en_core_web_sm')
    
    try:
        g_stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        g_stop_words = set(stopwords.words('english'))
    
    try:
        g_lemmatizer = WordNetLemmatizer()
    except LookupError:
        nltk.download('wordnet')
        g_lemmatizer = WordNetLemmatizer()

    _initiated = True

class ObfuscationTechniques:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.technique = config['technique']
        self.important_pos = ['NOUN', 'VERB', 'ADJ']
        
        self.sentence_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.leet_dict = self._initialize_leet_dict()

        self._idf_dict = None

        # Expanded invisible characters set
        self.invisible_chars = (
            '\u200B\u200C\u200D\u200E\u200F'  # Zero-width characters
            '\u2060\u2061\u2062\u2063\u2064'  # Word joiner and function application
            '\u206A\u206B\u206C\u206D\u206E\u206F'  # Inhibit symmetric swapping
            '\uFEFF'  # Zero-width no-break space
            '\u00AD'  # Soft hyphen
            '\u034F'  # Combining grapheme joiner
            + ''.join(chr(i) for i in range(32) if chr(i) not in '\n\r\t')  # Control characters excluding newline, carriage return, and tab
        )

        # Expanded similar characters set
        self.similar_chars = {
            'a': ['а', 'α', 'ä', 'ă', 'ā', 'ả', 'ạ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ὰ', 'ά', 'ἀ', 'ἁ', 'ἂ', 'ἃ', 'ἄ', 'ἅ', 'ἆ', 'ἇ'],
            'b': ['б', 'β', 'þ', 'ƀ', 'ɓ', 'ḃ', 'ḅ', 'ḇ', 'ƃ'],
            'c': ['ç', 'с', 'ć', 'ĉ', 'ċ', 'č', 'ƈ', 'ȼ', 'ḉ'],
            'd': ['д', 'δ', 'ð', 'ď', 'ḋ', 'ḍ', 'ḏ', 'ḑ', 'ḓ', 'ɗ'],
            'e': ['е', 'ε', 'é', 'è', 'ê', 'ë', 'ē', 'ĕ', 'ė', 'ę', 'ě', 'ȅ', 'ȇ', 'ḕ', 'ḗ', 'ḙ', 'ḛ', 'ḝ', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ'],
            'f': ['ф', 'ƒ', 'ḟ', 'ﬀ', 'ﬁ', 'ﬂ', 'ﬃ', 'ﬄ'],
            'g': ['г', 'γ', 'ğ', 'ĝ', 'ğ', 'ġ', 'ģ', 'ǥ', 'ǧ', 'ǵ', 'ḡ', 'ɠ'],
            'h': ['һ', 'ή', 'ħ', 'ĥ', 'ḣ', 'ḥ', 'ḧ', 'ḩ', 'ḫ', 'ẖ'],
            'i': ['і', 'ι', 'í', 'ì', 'î', 'ï', 'ĩ', 'ī', 'ĭ', 'į', 'ǐ', 'ȉ', 'ȋ', 'ḭ', 'ḯ', 'ỉ', 'ị'],
            'j': ['ј', 'ϳ', 'ĵ', 'ǰ', 'ȷ', 'ɉ'],
            'k': ['к', 'κ', 'ķ', 'ĸ', 'ǩ', 'ḱ', 'ḳ', 'ḵ', 'ƙ'],
            'l': ['ℓ', 'ł', 'ļ', 'ĺ', 'ļ', 'ľ', 'ḷ', 'ḹ', 'ḻ', 'ḽ', 'ƚ'],
            'm': ['м', 'μ', 'ṁ', 'ḿ', 'ṃ', 'ɱ'],
            'n': ['ñ', 'η', 'ń', 'ǹ', 'ņ', 'ň', 'ṅ', 'ṇ', 'ṉ', 'ṋ', 'ƞ'],
            'o': ['о', 'ο', 'ö', 'ò', 'ó', 'ô', 'õ', 'ō', 'ŏ', 'ő', 'ơ', 'ǒ', 'ǫ', 'ǭ', 'ȍ', 'ȏ', 'ȯ', 'ȱ', 'ṍ', 'ṏ', 'ṑ', 'ṓ', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ'],
            'p': ['р', 'ρ', 'þ', 'ṕ', 'ṗ', 'ƥ'],
            'q': ['ԛ', 'φ', 'ق', 'ɋ'],
            'r': ['г', 'ρ', 'ŕ', 'ŗ', 'ř', 'ȑ', 'ȓ', 'ṙ', 'ṛ', 'ṝ', 'ṟ', 'ɍ'],
            's': ['ѕ', 'ś', 'ş', 'ŝ', 'š', 'ș', 'ṡ', 'ṣ', 'ṥ', 'ṧ', 'ṩ', 'ƨ'],
            't': ['т', 'τ', 'ť', 'ţ', 'ț', 'ṫ', 'ṭ', 'ṯ', 'ṱ', 'ẗ', 'ƭ'],
            'u': ['υ', 'ц', 'ü', 'ù', 'ú', 'û', 'ũ', 'ū', 'ŭ', 'ů', 'ű', 'ų', 'ư', 'ǔ', 'ǖ', 'ǘ', 'ǚ', 'ǜ', 'ȕ', 'ȗ', 'ṳ', 'ṵ', 'ṷ', 'ṹ', 'ṻ', 'ụ', 'ủ', 'ứ', 'ừ', 'ử', 'ữ', 'ự'],
            'v': ['ν', 'ѵ', 'ṿ', 'ṽ', 'ⱱ'],
            'w': ['ѡ', 'ω', 'ŵ', 'ẁ', 'ẃ', 'ẅ', 'ẇ', 'ẉ', 'ẘ'],
            'x': ['х', 'χ', 'ж', 'ẋ', 'ẍ'],
            'y': ['у', 'γ', 'ÿ', 'ý', 'ỳ', 'ŷ', 'ȳ', 'ẏ', 'ỵ', 'ỷ', 'ỹ', 'ẙ'],
            'z': ['ż', 'ź', 'ž', 'ẑ', 'ẓ', 'ẕ', 'ƶ', 'ȥ']
        }

    def _initialize_leet_dict(self):
        return {
            'a': ['4', '@', '/-\\'], 'b': ['8', '|3', '|:'],
            'c': ['(', '{', '<'], 'd': ['|)', '|>', '<|'],
            'e': ['3', '&', '€'], 'f': ['|=', 'ph', '/='],
            'g': ['6', '9', '&'], 'h': ['#', '|-|', '}-{'],
            'i': ['1', '!', '|'], 'j': ['_|', '_/', '_7'],
            'k': ['|<', '|{', '|X'], 'l': ['|', '1', '|_'],
            'm': ['|v|', '|\\/|', '/\\/\\'], 'n': ['|\\|', '/\\/', '|/|'],
            'o': ['0', '()', '[]'], 'p': ['|*', '|>', '|"'],
            'q': ['0_', '0,', '(,)'], 'r': ['|2', '|?', '/2'],
            's': ['5', '$', 'z'], 't': ['7', '+', '-|-'],
            'u': ['|_|', '\\_\\', '/_/'], 'v': ['\\/', '|/', '\\/'],
            'w': ['\\/\\/', 'vv', '\\^/'], 'x': ['><', '}{', '><'],
            'y': ['`/', '¥', '\\|/'], 'z': ['2', '7_', '>_']
        }

    def set_idf_dict(self, idf_dict: Dict[str, float]):
        self._idf_dict = idf_dict
        self._max_idf = max(idf_dict.values())

    def obfuscate(self, text: str, level: float) -> str:
        """
        Obfuscate text
        Args:
            text: Input text
            base_level: Base obfuscation level (0-1)
        Returns:
            Obfuscated text
        """
        doc = g_nlp(text)
        words = [token.text for token in doc]
        
        # Calculate weight for each word
        word_weights = []
        if self._idf_dict is not None:
            # Use IDF weights
            max_idf = max(self._idf_dict.values())
            for word in words:
                if word.lower() in self._idf_dict:
                    weight = self._idf_dict[word.lower()] / max_idf
                else:
                    weight = 0.0
                word_weights.append(weight)
        else:
            # Use simple importance judgment
            for i, token in enumerate(doc):
                if token.ent_type_ or token.pos_ in self.important_pos:
                    word_weights.append(1.0)
                else:
                    word_weights.append(0.5)
        
        # Calculate actual obfuscation level for each word
        word_levels = [w * level for w in word_weights]
        
        if self.technique == 'character_swap':
            return self._character_swap(words, word_levels)
        elif self.technique == 'leet_speak':
            return self._leet_speak(words, word_levels)
        elif self.technique == 'similar_char_substitution':
            return self._similar_char_substitution(words, word_levels)
        elif self.technique == 'invisible_char_insertion':
            return self._invisible_char_insertion(words, word_levels)
        elif self.technique == 'synonym_replacement':
            return self._synonym_substitution(words, word_levels)
        else:
            raise ValueError(f"Unsupported obfuscation technique: {self.technique}")

    def _character_swap(self, words: List[str], word_levels: List[float]) -> str:
        """
        Character swap obfuscation
        Args:
            words: List of words
            word_levels: Obfuscation level for each word (0-1)
        """
        result = []
        for word, level in zip(words, word_levels):
            if len(word) < 2 or level == 0:
                result.append(word)
                continue
            
            chars = list(word)
            num_swaps = int(len(word) * level / 2)  # Each swap affects two characters, so divide by 2
            for _ in range(num_swaps):
                i, j = random.sample(range(len(word)), 2)
                chars[i], chars[j] = chars[j], chars[i]
            result.append(''.join(chars))
        
        return ' '.join(result)

    def _select_chars_for_substitution(self, words: List[str], word_levels: List[float], total_chars: int) -> List[Tuple[int, int]]:
        """
        Select character positions to replace based on each word's level
        Args:
            words: List of words
            word_levels: Obfuscation level for each word
            total_chars: Total number of characters to replace
        Returns:
            List of character positions to replace, each element is (word index, character index)
        """
        # Calculate character count multiplied by level
        weighted_lengths = [len(word) * level for word, level in zip(words, word_levels)]
        total_weighted_length = sum(weighted_lengths)
        
        if total_weighted_length == 0:
            return []
        
        # Normalize to get probability distribution
        word_probs = [w / total_weighted_length for w in weighted_lengths]
        
        # Draw word indices based on probability distribution
        selected_word_indices = np.random.choice(
            len(words), 
            size=total_chars, 
            p=word_probs
        )
        
        # For each selected word, randomly select a character position
        char_positions = []
        for word_idx in selected_word_indices:
            word = words[word_idx]
            if len(word) > 0:  # Ensure word is not empty
                char_idx = random.randrange(len(word))
                char_positions.append((word_idx, char_idx))
        
        return char_positions

    def _apply_char_substitution(self, words: List[str], word_levels: List[float], char_map: Dict[str, List[str]]) -> str:
        """
        Generic character substitution function
        Args:
            words: List of words
            word_levels: Obfuscation level for each word
            char_map: Character mapping dictionary
        Returns:
            Obfuscated text
        """
        # Calculate total number of characters to replace
        total_chars = sum(int(len(word) * level) for word, level in zip(words, word_levels))
        
        # Select character positions to replace
        char_positions = self._select_chars_for_substitution(words, word_levels, total_chars)
        
        # Apply substitution
        result = [list(word) for word in words]
        for word_idx, char_idx in char_positions:
            char = result[word_idx][char_idx].lower()
            if char in char_map:
                result[word_idx][char_idx] = random.choice(char_map[char])
        
        return ' '.join(''.join(chars) for chars in result)

    def _leet_speak(self, words: List[str], word_levels: List[float]) -> str:
        """
        Leet encoding obfuscation
        Args:
            words: List of words
            word_levels: Obfuscation level for each word (0-1)
        """
        return self._apply_char_substitution(words, word_levels, self.leet_dict)

    def _similar_char_substitution(self, words: List[str], word_levels: List[float]) -> str:
        """
        Similar character substitution
        Args:
            words: List of words
            word_levels: Obfuscation level for each word (0-1)
        """
        return self._apply_char_substitution(words, word_levels, self.similar_chars)

    def _synonym_substitution(self, words: List[str], word_levels: List[float]) -> str:
        """
        Synonym substitution
        Args:
            words: List of words
            word_levels: Obfuscation level for each word (0-1)
        Returns:
            Obfuscated text
        """
        result = list(words)
        
        for i, (word, level) in enumerate(zip(words, word_levels)):
            # Decide whether to replace the word based on level probability
            if random.random() >= level:
                continue
            
            synonyms = []
            # Get synonyms
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name().lower() != word.lower():
                        synonyms.append(lemma.name())
            
            # If synonyms are found, randomly select one to replace
            if synonyms:
                replacement = random.choice(synonyms).replace('_', ' ')
                result[i] = replacement
        
        return ' '.join(result)

class StringHelper:
    def __init__(self):
        self._sentence_model = None
        # self._model_name = 'distilbert-base-nli-mean-tokens'
        self._model_name = 'paraphrase-MiniLM-L6-v2'
        
    @property
    def sentence_model(self):
        """Lazy loading of sentence transformer model."""
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_model = SentenceTransformer(self._model_name)
            except ImportError as e:
                raise ImportError(f"Please install sentence-transformers: {e}")
        return self._sentence_model
    
    def set_idf_dict(self, idf_dict: Dict[str, float]):
        self._idf_dict = idf_dict
        self._max_idf = max(idf_dict.values())

    @property
    def idf_dict(self):
        return self._idf_dict
    
    @property
    def max_idf(self):
        return self._max_idf
    
    def clean_text(self, text: str, min_word_length: int = 2, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
        # Convert to lowercase
        text = text.lower()
        
        # Replace punctuation with spaces
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        
        # # Remove numbers
        # text = re.sub(r'\d+', '', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)

        # Tokenize
        words = word_tokenize(text)
        
        # Filter words
        stop_words = g_stop_words if remove_stopwords else set()
        words = [word for word in words 
                if len(word) >= min_word_length 
                and word not in stop_words]
        
        # Lemmatize
        if lemmatize:
            words = [g_lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)

    def preprocess_text(self, text: str, mode: str = 'lexical') -> Union[List[str], np.ndarray]:
        """
        Preprocess text based on specified mode.
        
        Args:
            text (str): Input text
            mode (str): 'lexical', 'idf', or 'semantic'
            
        Returns:
            Union[List[str], np.ndarray]: Preprocessed text as tokens or embedding
        """
        if mode == 'lexical':
            # Lexical preprocessing
            text = text.lower().replace('\\', ' ')      
            tokens = word_tokenize(text)
            # Remove stopwords and punctuation
            tokens = [token.translate(str.maketrans('', '', string.punctuation)) 
                     for token in tokens if token not in g_stop_words]
            return [t for t in tokens if t]
        elif mode == 'idf':
            # IDF preprocessing - more strict preprocessing than lexical
            assert hasattr(self, 'idf_dict'), "IDF dictionary not found, please set it first."
            # Disable lemmatization
            text = self.clean_text(text, lemmatize=False)
            tokens = word_tokenize(text)
            return [t for t in tokens if t]
        elif mode == 'semantic':
            # Semantic preprocessing - get embedding
            return self.sentence_model.encode(text)
        else:
            raise ValueError(f"Unsupported preprocessing mode: {mode}")

    def truncate_text(self, text: str, num_words: int) -> str:
        """Truncate text to a certain number of words."""
        tokens = word_tokenize(text)
        return ' '.join(tokens[:num_words])

    # Lexical similarities
    def lexical_jaccard_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate Jaccard similarity between token lists."""
        set1 = set(tokens1)
        set2 = set(tokens2)
        return len(set1 & set2) / len(set1 | set2) if set1 or set2 else 0.0

    def lexical_frequency_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate frequency-based similarity between token lists."""
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        common_words = set(freq1.keys()) & set(freq2.keys())
        
        if not common_words:
            return 0.0
            
        common_freq_sum = sum(min(freq1[word], freq2[word]) for word in common_words)
        total_freq_sum = max(sum(freq1.values()), sum(freq2.values()))
        return common_freq_sum / total_freq_sum if total_freq_sum > 0 else 0.0

    def lexical_sequence_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate sequence similarity (LCS-based) between token lists."""
        if not tokens1 or not tokens2:
            return 0.0
            
        m, n = len(tokens1), len(tokens2)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif tokens1[i-1] == tokens2[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        
        return L[m][n] / max(m, n)
    
    # IDF based similarities
    def idf_jaccard_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate IDF-weighted Jaccard similarity between token lists."""
        if not hasattr(self, 'idf_dict'):
            raise ValueError("IDF dictionary not found, please set it first.")
        
        set1 = set(tokens1)
        set2 = set(tokens2)
        common_words = set1 & set2
        idf_sum = 0
        for word in common_words:
            if word in self._idf_dict:
                idf_sum += self._idf_dict[word]
        idf_union = 0
        max_idf = max(self._idf_dict.values())
        for word in set1 | set2:
            if word in self._idf_dict:
                idf_union += self._idf_dict[word]
            # else:
            #     idf_union += max_idf
        return idf_sum / idf_union if idf_union > 0 else 0.0

    def idf_frequency_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate IDF-weighted frequency-based similarity between token lists."""
        if not hasattr(self, 'idf_dict'):
            raise ValueError("IDF dictionary not found, please set it first.")
        
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        common_words = set(freq1.keys()) & set(freq2.keys())
        
        if not common_words:
            return 0.0
        
        idf_sum = 0
        for word in common_words:
            if word in self._idf_dict:
                idf_sum += min(freq1[word], freq2[word]) * self._idf_dict[word]
        max_idf = max(self._idf_dict.values())
        freq1_sum = 0
        for word in freq1:
            if word in self._idf_dict:
                freq1_sum += freq1[word] * self._idf_dict[word]
            # else:
            #     freq1_sum += freq1[word] * max_idf
        freq2_sum = 0
        for word in freq2:
            if word in self._idf_dict:
                freq2_sum += freq2[word] * self._idf_dict[word]
            # else:
            #     freq2_sum += freq2[word] * max_idf
        total_freq_sum = max(freq1_sum, freq2_sum)
        return idf_sum / total_freq_sum if total_freq_sum > 0 else 0.0

    # Semantic similarities
    def semantic_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        return 1 - float(cosine(emb1, emb2))

    def semantic_euclidean_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate euclidean-based similarity between embeddings."""
        return float(1 / (1 + np.linalg.norm(emb1 - emb2)))

    def semantic_manhattan_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate Manhattan distance-based similarity between embeddings."""
        return float(1 / (1 + np.sum(np.abs(emb1 - emb2))))

    def semantic_dot_product_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate normalized dot product similarity between embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def calculate_overall_similarity_dict(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Calculate similarity metrics and return as dictionary.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            Dict[str, float]: Dictionary containing similarity scores
        """

        similarities = {}

        # 1. In lexical mode
        mode = 'lexical'
        data1 = self.preprocess_text(text1, mode)
        data2 = self.preprocess_text(text2, mode)
        similarities['jaccard'] = self.lexical_jaccard_similarity(data1, data2)
        similarities['frequency'] = self.lexical_frequency_similarity(data1, data2)
        similarities['sequence'] = self.lexical_sequence_similarity(data1, data2)

        # 2. In IDF enabled mode
        mode = 'idf'
        if hasattr(self, 'idf_dict'):
            data1 = self.preprocess_text(text1, mode)
            data2 = self.preprocess_text(text2, mode)
            similarities['ijaccard'] = self.idf_jaccard_similarity(data1, data2)
            similarities['ifrequency'] = self.idf_frequency_similarity(data1, data2)

        # 3. In semantic mode
        mode = 'semantic'
        data1 = self.preprocess_text(text1, mode)
        data2 = self.preprocess_text(text2, mode)
        similarities['cosine'] = self.semantic_cosine_similarity(data1, data2)
        similarities['euclidean'] = self.semantic_euclidean_similarity(data1, data2)
        similarities['manhattan'] = self.semantic_manhattan_similarity(data1, data2)
        
        return similarities

if __name__ == "__main__":
    # Test text
    test_text = "The quick brown fox jumps over the lazy dog."
    print(f"Original text: {test_text}\n")

    # Configuration
    config = {
        'technique': 'character_swap',
        'nlp_dataset': 'en_core_web_md',
    }

    # Create ObfuscationTechniques instance
    obfuscator = ObfuscationTechniques(config)

    # Test character_swap
    print("Character Swap Obfuscation:")
    for level in np.linspace(0, 1, 5):
        obfuscated = obfuscator.obfuscate(test_text, level)
        print(f"Level {level}: {obfuscated}")
    print()

    # Test word_shuffle
    config['technique'] = 'word_shuffle'
    obfuscator = ObfuscationTechniques(config)
    print("Word Shuffle Obfuscation:")
    for level in np.linspace(0, 1, 5):
        obfuscated = obfuscator.obfuscate(test_text, level)
        print(f"Level {level}: {obfuscated}")
    print()

    # Test synonym_replacement
    config['technique'] = 'synonym_replacement'
    obfuscator = ObfuscationTechniques(config)
    print("Synonym Replacement Obfuscation:")
    for level in np.linspace(0, 1, 5):
        obfuscated = obfuscator.obfuscate(test_text, level)
        print(f"Level {level}: {obfuscated}")
    print()

    # Test leet_speak
    config['technique'] = 'leet_speak'
    obfuscator = ObfuscationTechniques(config)
    print("Leet Speak Obfuscation:")
    for level in np.linspace(0, 1, 5):
        obfuscated = obfuscator.obfuscate(test_text, level)
        print(f"Level {level}: {obfuscated}")
    print()

    # Test evaluate_response
    original_label = "Animal"
    responses = [
        "This sentence is about an animal.",
        "The text describes a quick movement.",
        "This is a common English pangram.",
    ]
    print("Evaluate Response:")
    for response in responses:
        result = obfuscator.evaluate_response(response, test_text, original_label)
        print(f"Response: '{response}'\nEvaluation: {'Correct' if result else 'Incorrect'}")