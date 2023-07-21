from .base_analyzer import BaseAnalyzer
from .utils_ner_basic import LinkedEntity, SentenceNerOutput
from .link_util import punctuation_pat
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EntityFilter(BaseAnalyzer):
    """
    docstring
    """
    def __init__(self, entity_types, database_id_name, high_levenshtein_score=95) -> None:
        super().__init__(entity_types, database_id_name, high_levenshtein_score)

    def filter_entity(self, sentence_output:SentenceNerOutput):
        """  Only keep full linked entity
        returns: a list [entity_type, start_index, end_index, phrase, linked_phrase, database_id]

        TODO the link is hard to judge as fully correct by rule, as we had better not treat 'cascade' as stop words.
            "entity": {
                "text": "caspase - dependent apoptosis",
                "type": "PATHWAY",
                "start_char_index": 13,
                "end_char_index": 40
            },
            "linked_name": "caspase dependent cascade in apoptosis",
            "distance": 0.18603073060512543,
            "extra_info": {
                "source": "PID:319"
            }
        """
        entities = sentence_output.entities
        sentence_text = sentence_output.sentence_text
        results = []
        for item in entities:
            entity = item.entity
            ner_phrase = entity.text
            linked_phrase = item.linked_name
            distance = item.distance
            database_id = item.extra_info[self.database_id_name]
            ner_phrase_lower = punctuation_pat.sub(' ', ner_phrase.lower())
            if self.text_normalizer:
                ner_phrase_lower = self.text_normalizer.normalize(ner_phrase_lower)
            linked_phrase_lower = punctuation_pat.sub(' ', linked_phrase.lower())


            is_safe = "0"
            if self.is_full_correct(ner_phrase_lower, linked_phrase_lower):
                is_safe = "1"
                
            start_index = entity.start_char_index
            end_index = entity.end_char_index
            phrase = sentence_text[start_index: end_index]
            results.append([entity.type, str(start_index), str(
                end_index), phrase, linked_phrase, database_id, str(distance), is_safe])
            # else:
            #     results.append([entity.type, str(start_index), str(
            #         end_index), phrase, linked_phrase, database_id, str(distance)])
        return results
