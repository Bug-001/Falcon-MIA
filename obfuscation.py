import re
import random
import string
from typing import Dict, Any, List
import nltk
from nltk.corpus import wordnet
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import spacy
from collections import defaultdict

class ObfuscationTechniques:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.technique = config.get('technique', 'character_swap')
        self.nltk_downloaded = False
        self.nlp_dataset = config.get('nlp_dataset', 'en_core_web_sm')
        self.important_pos = ['NOUN', 'VERB', 'ADJ', "PRON"]
        try:
            self.nlp = spacy.load(self.nlp_dataset)
        except OSError:
            print("Downloading English language model...")
            from subprocess import check_call
            check_call(["python", "-m", "spacy", "download", self.nlp_dataset])
            self.nlp = spacy.load(self.nlp_dataset)
        self.sentence_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.keyword_threshold = config.get('keyword_threshold', 0.3)
        self.leet_dict = {
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
        doc = self.nlp(text)
        words = [token.text for token in doc]
        important_words = [token.text for token in doc if token.ent_type_ or token.pos_ in ['NOUN', 'VERB', 'ADJ']]
        
        num_words_to_obfuscate = round(len(important_words) * level)
        words_to_obfuscate = set(random.sample(important_words, num_words_to_obfuscate))
        
        obfuscated_words = []
        for word in words:
            if word in words_to_obfuscate:
                obfuscated_words.append(self._obfuscate_word(word, 1.0))  # 全部混淆
            else:
                obfuscated_words.append(self._obfuscate_word(word, level))  # 部分混淆
        
        return ' '.join(obfuscated_words)

    def _obfuscate_word(self, word: str, level: float) -> str:
        if self.technique == 'character_swap':
            return self._character_swap(word, level)
        elif self.technique == 'leet_speak':
            return self._leet_speak(word, level)
        elif self.technique == 'word_shuffle':
            return self._word_shuffle(word, level)
        else:
            return word

    def _character_swap(self, word: str, level: float) -> str:
        if len(word) < 2:
            return word
        chars = list(word)
        num_swaps = int(len(word) * level / 2)  # 每次交换影响两个字符
        for _ in range(num_swaps):
            i, j = random.sample(range(len(word)), 2)
            chars[i], chars[j] = chars[j], chars[i]
        return ''.join(chars)

    def _word_shuffle(self, word: str, level: float) -> str:
        if len(word) < 2:
            return word
        chars = list(word)
        num_shuffles = int(len(word) * level)
        for _ in range(num_shuffles):
            i = random.randint(0, len(chars) - 1)
            j = random.randint(0, len(chars) - 1)
            chars[i], chars[j] = chars[j], chars[i]
        return ''.join(chars)

    def _leet_speak(self, word: str, level: float) -> str:
        chars = list(word)
        num_changes = round(len(word) * level)
        for _ in range(num_changes):
            i = random.randint(0, len(chars) - 1)
            char = chars[i].lower()
            if char in self.leet_dict:
                chars[i] = random.choice(self.leet_dict[char])
        return ''.join(chars)

    def _synonym_replacement(self, word: str) -> str:
        if not self.nltk_downloaded:
            import nltk
            nltk.download('wordnet', quiet=True)
            self.nltk_downloaded = True

        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name().lower() != word.lower():
                    synonyms.append(lemma.name())
        
        if synonyms:
            return random.choice(synonyms)
        return word

    def _leet_speak(self, word: str, level: float) -> str:
        chars = list(word)
        num_changes = int(len(word) * level)
        for _ in range(num_changes):
            i = random.randint(0, len(chars) - 1)
            char = chars[i].lower()
            if char in self.leet_dict:
                chars[i] = random.choice(self.leet_dict[char])
        return ''.join(chars)

    def evaluate_response(self, response: str, original_text: str, original_label: str) -> bool:
        # 使用 spaCy 进行命名实体识别和词性标注
        original_doc = self.nlp(original_text)
        response_doc = self.nlp(response)

        # 提取原文中的重要实体和关键词
        original_entities = set([ent.text.lower() for ent in original_doc.ents])
        original_keywords = set([token.text.lower() for token in original_doc if token.pos_ in self.important_pos])

        # 提取响应中的实体和关键词
        response_entities = set([ent.text.lower() for ent in response_doc.ents])
        response_keywords = set([token.text.lower() for token in response_doc if token.pos_ in self.important_pos])

        # 计算实体和关键词的匹配程度
        entity_match_ratio = len(original_entities.intersection(response_entities)) / len(original_entities) if original_entities else 0
        keyword_match_ratio = len(original_keywords.intersection(response_keywords)) / len(original_keywords) if original_keywords else 0

        # 使用 spaCy 的相似度函数计算语义相似度
        semantic_similarity = original_doc.similarity(response_doc)

        # 检查原始标签是否出现在响应中
        label_match = original_label.lower() in response.lower()

        # 设置阈值
        entity_threshold = 0.5
        keyword_threshold = 0.3
        similarity_threshold = 0.7

        # 如果满足任一条件，认为模型识别了该例子
        return (entity_match_ratio > entity_threshold or
                keyword_match_ratio > keyword_threshold or
                semantic_similarity > similarity_threshold or
                label_match)

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