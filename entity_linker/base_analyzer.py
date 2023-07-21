from typing import List
from nltk.stem import WordNetLemmatizer

from rapidfuzz.fuzz import ratio
from .utils_ner_basic import SentenceNerOutput
from .link_util import read_stop_words, read_entity_name_matcher, ResultItem, ResultItems, is_word_in_words
from .link_util import punctuation_pat, read_kept_words
# from projects.pathway.pathway_phrase_normalizer import PathwayNormalizer
# from .link_preprocess.pathway_link_preprocess import PathwayLinkProcessor
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


LEAST_CROSS_INCLUDED_LEN = 3


class BaseAnalyzer():
    """
    lemma tool choose
    # spacy lemma has pre process pipline to auto recoginize pos, much lower speed and not good results as nltk in general.
    # spacy treats "blood dyscrasias" as "blood dyscrasias", while nltk as "blood dyscrasias" to del the ending "s".
    # spacy treats "hypercalcemias" as "hypercalcemias", while nltk as "hypercalcemia" to del the ending "s".

    Histone acetylation is regulated by a balance between the activity of histone acetyltransferases ( HATs ) and histone deacetylases (HDACs).
    Histone deacetylases (EC 3.5.1.98, HDAC) are a class of enzymes that remove acetyl groups (O=C-CH3) from an ε-N-acetyl lysine amino acid on a histone, allowing the histones to wrap the DNA more tightly.[2] This is important because DNA is wrapped around histones, and DNA expression is regulated by acetylation and de-acetylation. Its action is opposite to that of histone acetyltransferase. HDAC proteins are now also called lysine deacetylases (KDAC), to describe their function rather than their target, which also includes non-histone proteins.
    spacy treats "hdacs" as "hdac", while nltk as "hdacs" not to del the ending "s". Here spacy does better, here "s" is a plural.

    # In short, precison difference is very small, but speed difference is large.
    As NER is mostly noun, So choose nltk lemma.
    """
    def __init__(self,
        entity_types,
        database_id_name,
        high_levenshtein_score = 95,
    ) -> None:
        self.high_levenshtein_score = high_levenshtein_score
        self.lemmatizer = WordNetLemmatizer()
        # entity_types_str is pre lower() case
        self.entity_types_str = '-'.join(entity_types)
        self.stop_words = read_stop_words(entity_types)
        self.kept_words = read_kept_words(entity_types)
        logger.debug(f'self.stop_words: {self.stop_words}')
        self.entity_name_matcher = read_entity_name_matcher(entity_types)
        self.database_id_name = database_id_name        
        self.text_normalizer = None
        self.preprocessor = None
        # if self.entity_types_str == 'pathway':
        #     self.text_normalizer = PathwayNormalizer
        #     self.link_preprocessor = PathwayLinkProcessor

    def basic_analyze(self, results: List[SentenceNerOutput]):
        """ total_num excludes empty_entities number.
        a and b are 2 strings.
        full_identical, a = b.
        cross_included, a in b or b in a.
        """
        full_correct_items = []
        cross_included_items = []
        high_similar_items = []  # levenshtein_score >= high_levenshtein_score
        empty_items = []
        abnormal_ner_items = []
        other_items = []

        self.total_num = 0
        for sentence_ner_output in results:
            entities = sentence_ner_output.entities
            sentence = sentence_ner_output.sentence_text
            if not entities:
                empty_items.append(ResultItem(sentence_text=sentence))
                continue
            for item in entities:
                ner_phrase = item.entity.text
                linked_phrase = item.linked_name
                linked_id = item.extra_info.get(self.database_id_name, '')
                distance = item.distance
                entity_type = item.entity.type

                self.total_num += 1
                if self.total_num < 2:
                    logger.info(f'sentence_items examples:\n{sentence_ner_output}')
                ner_phrase_lower = punctuation_pat.sub(' ', ner_phrase.lower())
                if self.text_normalizer:
                    ner_phrase_lower = self.text_normalizer.normalize(ner_phrase_lower)
                linked_phrase_lower = punctuation_pat.sub(' ', linked_phrase.lower())

                if self.is_ignored_entity(ner_phrase_lower):
                    other_items.append(
                        ResultItem(ner_phrase, linked_phrase, entity_type, linked_id, distance, -1, sentence))
                    continue

                if ner_phrase_lower.strip() == '':
                    abnormal_ner_items.append(
                        ResultItem(ner_phrase, linked_phrase, entity_type, linked_id, distance, l_score, sentence))
                    continue

                if self.is_full_correct(ner_phrase_lower, linked_phrase_lower):
                    full_correct_items.append(
                        ResultItem(ner_phrase, linked_phrase, entity_type, linked_id, 0, -1, sentence))
                elif (self.is_word_cross(ner_phrase_lower, linked_phrase_lower)
                    or self.is_upper_chars_same(ner_phrase, linked_phrase)):
                    cross_included_items.append(
                        ResultItem(ner_phrase, linked_phrase, entity_type, linked_id, distance, -1, sentence))
                else:
                    l_score = ratio(ner_phrase_lower, linked_phrase_lower)
                    if l_score >= self.high_levenshtein_score:
                        high_similar_items.append(
                            ResultItem(ner_phrase, linked_phrase, entity_type, linked_id, distance, l_score, sentence))
                    elif distance == 0:
                        abnormal_ner_items.append(
                            ResultItem(ner_phrase, linked_phrase, entity_type, linked_id, distance, l_score, sentence))
                    else:
                        other_items.append(
                            ResultItem(ner_phrase, linked_phrase, entity_type, linked_id, distance, l_score, sentence))
        self.result = {
            'full_correct_items': self.get_items('full_correct_items', full_correct_items),
            'cross_included_items': self.get_items('cross_included_items', cross_included_items),
            'high_similar_items': self.get_items('high_similar_items', high_similar_items),
            'abnormal_ner_items': self.get_items('abnormal_ner_items', abnormal_ner_items),
            'other_items': self.get_items('other_items', other_items),
            'empty_items': self.get_items('empty_items', empty_items),
        }

        self.result_summary = {'entity_types': self.entity_types_str, 'total_number(excluding empty items)': self.total_num}
        all_high_similar_num = 0
        all_high_similar_types = ['full_correct_items', 'cross_included_items', 'high_similar_items']
        for k, v in self.result.items():
            self.result_summary[k] = f'num: {v.num}, ratio: {v.ratio}'
            if k in all_high_similar_types:
                all_high_similar_num += v.num
        self.result_summary['all_high_similar_num'] = all_high_similar_num
        self.result_summary['all_high_similar_ratio'] = round(all_high_similar_num / self.total_num, 4)

    def is_ignored_entity(self, ner_phrase_lower):
        """ Some phrase is too general and in the stop_words, e.g. 'disease' """
        if ner_phrase_lower in self.stop_words and ner_phrase_lower not in self.kept_words:
            return True
        return False

    def is_full_correct(self, ner_phrase_lower:str, linked_phrase_lower:str):
        """ 3 situations are treated as full correct
        There is risk to judge plural by single s
        Del this judge: such as "ts" vs "tss". Situation: if only extra "s" at end in one entity, treat the two entities the same. IF add this judge, should also add extra limit, such as full levenshtein ratio >=80, unworth.
        As such situation is very rare, 1200~1600， only 1 instance, 0.1% difference.
        if f'{s1_no_punctuation}s' == s2_no_punctuation or f'{s2_no_punctuation}s' == s1_no_punctuation:
            return True
        """
        # logger.debug(f'ner_phrase_lower {ner_phrase_lower} vs {linked_phrase_lower}')
        if ner_phrase_lower == linked_phrase_lower:
            return True
        if punctuation_pat.sub('', ner_phrase_lower) == punctuation_pat.sub('', linked_phrase_lower):
            return True
        if self.entity_name_matcher and self.entity_name_matcher.get(ner_phrase_lower, '') == linked_phrase_lower:
            return True
        words_ner = [self.lemmatizer.lemmatize(word) for word in ner_phrase_lower.split()]
        words_link = [self.lemmatizer.lemmatize(word) for word in linked_phrase_lower.split()]
        # logger.debug(f'{words_1} vs {words_2}')
        words_ner = set(words_ner)
        words_link = set(words_link)
        if self.entity_types_str == 'pathway':
            if self.link_preprocessor.is_not_full_correct(words_ner, words_link):
                return False
        for word in self.stop_words:
            words_ner.discard(word)
            words_link.discard(word)
        if words_ner == words_link:
            return True
        return False

    def is_word_cross(self, sentence_1, sentence_2):
        """ Not consider levenshtein_score """
        words_1 = sentence_1.split()
        words_2 = sentence_2.split()
        if len(words_1) > len(words_2):
            temp = words_2
            words_2 = words_1
            words_1 = temp
        words_2 = [self.lemmatizer.lemmatize(word) for word in words_2]
        for word in words_1:
            word = self.lemmatizer.lemmatize(word)
            if not is_word_in_words(word, words_2):
                return False
        return True

    def is_cross_included(self, ner_phrase_lower:str, linked_phrase_lower:str):
        """
        The either ner entity and linked entity must have length >= LEAST_CROSS_INCLUDED_LEN
        """
        if len(ner_phrase_lower) < LEAST_CROSS_INCLUDED_LEN or len(linked_phrase_lower) < LEAST_CROSS_INCLUDED_LEN:
            return False
        if ner_phrase_lower in linked_phrase_lower or linked_phrase_lower in ner_phrase_lower:
            return True
        if self.is_word_cross(ner_phrase_lower, linked_phrase_lower):
            return True
        return False

    def is_upper_chars_same(self, ner_entity:str, linked_entity:str):
        """ Ataxia-Telangiectasia vs A - T, check the first upper char and compare equality. """
        upper_chars_ner = []
        upper_chars_target = []
        for char in ner_entity:
            if 'A' <= char <= 'Z':
                upper_chars_ner.append(char)
        for char in linked_entity:
            if 'A' <= char <= 'Z':
                upper_chars_target.append(char)

        if len(upper_chars_ner) == len(upper_chars_target) and len(upper_chars_ner) > 1:
            for char1, char2 in zip(upper_chars_ner, upper_chars_target):
                if char1 != char2:
                    return False
            return True
        return False

    def get_items(self, items_type, items):
        return ResultItems(items_type, len(items), round(len(items) / self.total_num, 4), items)
