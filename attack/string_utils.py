import re
import math
import random
import string
from typing import Dict, Any, List, Set, Optional, Tuple, Union
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import spacy
from collections import defaultdict, Counter

class ObfuscationTechniques:
    _nlp_initialized = False
    nlp = None
    stop_words = None

    @classmethod
    def _load_nlp_resources(cls, nlp_dataset):
        try:
            cls.nlp = spacy.load(nlp_dataset)
        except OSError:
            print("Downloading English language model...")
            from subprocess import check_call
            check_call(["python", "-m", "spacy", "download", nlp_dataset])
            cls.nlp = spacy.load(nlp_dataset)
        
        try:
            cls.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            cls.stop_words = set(stopwords.words('english'))

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.technique = config.get('technique', 'character_swap')
        self.nltk_downloaded = False
        self.nlp_dataset = config.get('nlp_dataset', 'en_core_web_sm')
        self.important_pos = ['NOUN', 'VERB', 'ADJ', "PRON"]
        
        # Load NLP models and resources
        if not ObfuscationTechniques._nlp_initialized:
            ObfuscationTechniques._load_nlp_resources(self.nlp_dataset)
            ObfuscationTechniques._nlp_initialized = True
        
        self.sentence_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.keyword_threshold = config.get('keyword_threshold', 0.3)
        self.leet_dict = self._initialize_leet_dict()

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

    def obfuscate(self, text: str, level: float) -> str:
        doc = ObfuscationTechniques.nlp(text)
        words = [token.text for token in doc]
        important_indices = [i for i, token in enumerate(doc) if token.ent_type_ or token.pos_ in self.important_pos]
        
        num_words_to_obfuscate = round(len(important_indices) * level)
        indices_to_obfuscate = set(random.sample(important_indices, num_words_to_obfuscate))
        
        if self.technique == 'character_swap':
            return self._character_swap(words, indices_to_obfuscate, level)
        elif self.technique == 'leet_speak':
            return self._leet_speak(words, indices_to_obfuscate, level)
        elif self.technique == 'word_shuffle':
            return self._word_shuffle(words, indices_to_obfuscate, level)
        elif self.technique == 'similar_char_substitution':
            return self._similar_char_substitution(words, indices_to_obfuscate, level)
        elif self.technique == 'invisible_char_insertion':
            return self._invisible_char_insertion(words, indices_to_obfuscate, level)
        elif self.technique == 'synonym_replacement':
            return self._synonym_substitution(words, indices_to_obfuscate, level)
        else:
            return text

    def _character_swap(self, words: List[str], indices_to_obfuscate: set, level: float) -> str:
        for i in indices_to_obfuscate:
            words[i] = self._swap_characters(words[i], level)
        return ' '.join(words)

    def _swap_characters(self, word: str, level: float) -> str:
        if len(word) < 2:
            return word
        chars = list(word)
        num_swaps = int(len(word) * level / 2)
        for _ in range(num_swaps):
            i, j = random.sample(range(len(word)), 2)
            chars[i], chars[j] = chars[j], chars[i]
        return ''.join(chars)

    def _select_chars_to_obfuscate(self, words: List[str], num_chars: int, word_indices: Optional[Set[int]] = None, allow_repeat: bool = False) -> List[Tuple[int, int]]:
        char_positions = []
        for word_idx, word in enumerate(words):
            if word_indices is None or word_idx in word_indices:
                char_positions.extend((word_idx, char_idx) for char_idx, char in enumerate(word) if char.isalpha())
        
        if allow_repeat:
            return [random.choice(char_positions) for _ in range(num_chars)]
        else:
            return random.sample(char_positions, min(num_chars, len(char_positions)))

    def _leet_speak(self, words: List[str], important_indices: Set[int], level: float) -> str:
        # Calculate characters to change for important and non-important words
        important_chars = sum(len(word) for i, word in enumerate(words) if i in important_indices and word.isalpha())
        non_important_chars = sum(len(word) for i, word in enumerate(words) if i not in important_indices and word.isalpha())
        
        chars_to_change_important = int(important_chars * level)
        chars_to_change_non_important = int(non_important_chars * level / 2)

        # Select characters to obfuscate for important and non-important words
        important_chars_to_obfuscate = self._select_chars_to_obfuscate(words, chars_to_change_important, important_indices)
        non_important_chars_to_obfuscate = self._select_chars_to_obfuscate(words, chars_to_change_non_important, set(range(len(words))) - important_indices)

        # Combine the lists of characters to obfuscate
        all_chars_to_obfuscate = important_chars_to_obfuscate + non_important_chars_to_obfuscate

        # Perform the leet speak obfuscation
        words_list = list(words)  # Convert to list to allow item assignment
        for word_idx, char_idx in all_chars_to_obfuscate:
            char = words_list[word_idx][char_idx].lower()
            if char in self.leet_dict:
                word = list(words_list[word_idx])
                word[char_idx] = random.choice(self.leet_dict[char])
                words_list[word_idx] = ''.join(word)

        return ' '.join(words_list)

    def _synonym_substitution(self, words: List[str], important_indices: Set[int], level: float) -> List[str]:
        num_to_change = int(len(important_indices) * level)
        indices_to_change = random.sample(list(important_indices), min(num_to_change, len(important_indices)))
        
        words_list = list(words)
        for idx in indices_to_change:
            synonyms = []
            for syn in wordnet.synsets(words_list[idx]):
                for lemma in syn.lemmas():
                    if lemma.name().lower() != words_list[idx].lower():
                        synonyms.append(lemma.name())
            if synonyms:
                words_list[idx] = random.choice(synonyms).replace('_', ' ')
        
        return ' '.join(words_list)

    def _invisible_char_insertion(self, words: List[str], important_indices: Set[int], level: float) -> str:
        total_chars = sum(len(word) for word in words if word.isalpha())
        chars_to_insert_important = int(total_chars * level)
        chars_to_insert_non_important = int(total_chars * level / 2)

        important_positions = self._select_chars_to_obfuscate(words, chars_to_insert_important, important_indices, allow_repeat=True)
        non_important_positions = self._select_chars_to_obfuscate(words, chars_to_insert_non_important, set(range(len(words))) - important_indices, allow_repeat=True)

        all_positions = important_positions + non_important_positions
        words_list = list(words)

        for word_idx, char_idx in all_positions:
            invisible_char = random.choice(self.invisible_chars)
            word = list(words_list[word_idx])
            word.insert(char_idx, invisible_char)
            words_list[word_idx] = ''.join(word)

        return ' '.join(words_list)

    def _similar_char_substitution(self, words: List[str], important_indices: Set[int], level: float) -> str:
        total_chars = sum(len(word) for word in words if word.isalpha())
        chars_to_change_important = int(total_chars * level)
        chars_to_change_non_important = int(total_chars * level / 2)

        important_chars_to_obfuscate = self._select_chars_to_obfuscate(words, chars_to_change_important, important_indices)
        non_important_chars_to_obfuscate = self._select_chars_to_obfuscate(words, chars_to_change_non_important, set(range(len(words))) - important_indices)

        all_chars_to_obfuscate = important_chars_to_obfuscate + non_important_chars_to_obfuscate

        words_list = list(words)
        for word_idx, char_idx in all_chars_to_obfuscate:
            char = words_list[word_idx][char_idx].lower()
            if char in self.similar_chars:
                word = list(words_list[word_idx])
                word[char_idx] = random.choice(self.similar_chars[char])
                words_list[word_idx] = ''.join(word)

        return ' '.join(words_list)

    def _word_shuffle(self, words: List[str], indices_to_obfuscate: set, level: float) -> str:
        for i in indices_to_obfuscate:
            words[i] = self._shuffle_word(words[i], level)
        return ' '.join(words)

    def _shuffle_word(self, word: str, level: float) -> str:
        if len(word) < 2:
            return word
        chars = list(word)
        num_shuffles = int(len(word) * level)
        for _ in range(num_shuffles):
            i, j = random.sample(range(len(chars)), 2)
            chars[i], chars[j] = chars[j], chars[i]
        return ''.join(chars)

class StringHelper:
    def __init__(self):
        self._stop_words = None
        self._sentence_model = None
        # self._model_name = 'distilbert-base-nli-mean-tokens'
        self._model_name = 'paraphrase-MiniLM-L6-v2'
        
    @property
    def stop_words(self):
        """Lazy loading of NLTK stop words."""
        if self._stop_words is None:
            try:
                import nltk
                from nltk.corpus import stopwords
                self._stop_words = set(stopwords.words('english'))
            except LookupError as e:
                print(f"Error loading NLTK resources: {e}")
                self._stop_words = set()
        return self._stop_words
        
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

    def preprocess_text(self, text: str, mode: str = 'lexical') -> Union[List[str], np.ndarray]:
        """
        Preprocess text based on specified mode.
        
        Args:
            text (str): Input text
            mode (str): 'lexical' or 'semantic'
            
        Returns:
            Union[List[str], np.ndarray]: Preprocessed text as tokens or embedding
        """
        if mode == 'lexical':
            # Lexical preprocessing
            text = text.lower()
            tokens = word_tokenize(text)
            # Remove stopwords and punctuation
            tokens = [token.translate(str.maketrans('', '', string.punctuation)) 
                     for token in tokens if token not in ObfuscationTechniques.stop_words]
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

    # Semantic similarities
    def semantic_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        return float(1 - cosine(emb1, emb2))

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

        # 2. In semantic mode
        mode = 'semantic'
        data1 = self.preprocess_text(text1, mode)
        data2 = self.preprocess_text(text2, mode)
        similarities['cosine'] = self.semantic_cosine_similarity(data1, data2)
        similarities['euclidean'] = self.semantic_euclidean_similarity(data1, data2)
        similarities['manhattan'] = self.semantic_manhattan_similarity(data1, data2)
        similarities['dot_product'] = self.semantic_dot_product_similarity(data1, data2)
        
        return similarities

if __name__ == "__main__":
    # 测试文本
    test_text = "The quick brown fox jumps over the lazy dog."
    print(f"Original text: {test_text}\n")

    # 配置
    config = {
        'technique': 'character_swap',
        'nlp_dataset': 'en_core_web_md',
    }

    # 创建ObfuscationTechniques实例
    obfuscator = ObfuscationTechniques(config)

    # 测试character_swap
    print("Character Swap Obfuscation:")
    for level in np.linspace(0, 1, 5):
        obfuscated = obfuscator.obfuscate(test_text, level)
        print(f"Level {level}: {obfuscated}")
    print()

    # 测试word_shuffle
    config['technique'] = 'word_shuffle'
    obfuscator = ObfuscationTechniques(config)
    print("Word Shuffle Obfuscation:")
    for level in np.linspace(0, 1, 5):
        obfuscated = obfuscator.obfuscate(test_text, level)
        print(f"Level {level}: {obfuscated}")
    print()

    # 测试synonym_replacement
    config['technique'] = 'synonym_replacement'
    obfuscator = ObfuscationTechniques(config)
    print("Synonym Replacement Obfuscation:")
    for level in np.linspace(0, 1, 5):
        obfuscated = obfuscator.obfuscate(test_text, level)
        print(f"Level {level}: {obfuscated}")
    print()

    # 测试leet_speak
    config['technique'] = 'leet_speak'
    obfuscator = ObfuscationTechniques(config)
    print("Leet Speak Obfuscation:")
    for level in np.linspace(0, 1, 5):
        obfuscated = obfuscator.obfuscate(test_text, level)
        print(f"Level {level}: {obfuscated}")
    print()

    # 测试evaluate_response
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