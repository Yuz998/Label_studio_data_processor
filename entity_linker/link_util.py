from dataclasses import dataclass
import json
from pathlib import Path
from typing import List
import re
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


punctuation_pat = re.compile(r'[\W_]+')


@dataclass
class ResultItem():
    ner_entity: str = ''
    linked_entity: str = ''
    entity_type: str = ''
    linked_id: str = ''
    distance: float = -1
    levenshtein_score: float = -1
    sentence_text: str = ''

    def __str__(self) -> str:
        return (
            f'NER_entity   : {self.ner_entity}\nLinked_entity: {self.linked_entity}\nEntity_type: {self.entity_type}\n'
            f'Levenshtein_score: {self.levenshtein_score}, Distance: {self.distance}. Sentence(below):\n'
            f'{self.sentence_text}'
        )


@dataclass
class ResultItems():
    items_type: str
    num: int
    ratio: float
    details: List[ResultItem]

    def __str__(self) -> str:
        return f'item_type: {self.items_type}, num: {self.num}, ratio: {self.ratio}'


def is_word_in_words(input_word, words):
    """ Not consider levenshtein_score """
    for word in words:
        if input_word == word:
            return True
    return False


def read_stop_words(entity_types:List[str]):
    """ Some entity names are too general and so ignored, such as 'injury' in disease type. """
    rootpath = 'configs/stop_words'
    ignored_entity_names = []
    for file in Path(rootpath).glob('*.txt'):
        entity_type = file.stem.split('_')[0]
        if 'stop' not in file.stem: continue
        if entity_type in entity_types:
            logger.info(f'Read stop words from {file}')
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    ignored_entity_names.append(line.strip())
    return ignored_entity_names

def read_kept_words(entity_types:List[str]):
    """ Some entity names too general entity names, such as 'transcription'.
    They are kept entity names if it only one word; but when combine with others, it can be treated as stop words. """
    rootpath = 'configs/stop_words'
    kept_entity_names = []
    for file in Path(rootpath).glob('*.txt'):
        entity_type = file.stem.split('_')[0]
        if 'kept' not in file.stem: continue
        if entity_type in entity_types:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    kept_entity_names.append(line.strip())
    return kept_entity_names


def read_entity_name_matcher(entity_types):
    rootpath = 'configs/entity_name_matcher'
    entity_name_matcher = {}
    for file in Path(rootpath).glob('*.json'):
        entity_type = file.stem.split('_')[0]
        if entity_type in entity_types:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entity_name_matcher.update(data)
    return entity_name_matcher
