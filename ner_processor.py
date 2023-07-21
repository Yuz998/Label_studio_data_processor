import re
from model.utils_ner_basic import SentenceNerOutput, Entity


punctuations_at_end = re.compile(f'[\W_]+$')


class NerPostprocessor():
    """ TODO add_ahead()
    """

    @classmethod
    def ner_postprocess(cls, ner_outs):
        # cls.merge_next(ner_outs)
        # cls.remove_punctuations_at_head_end(ner_outs)
        return ner_outs

    @classmethod
    def merge_next(cls, ner_outs):
        """ 
        """
        for sent_ner_out in ner_outs:
            new_entities = []
            len_entities= len(sent_ner_out.entities)
            i = 0
            while i < len_entities:
                entity = sent_ner_out.entities[i]
                if i + 1 < len_entities and entity.end_char_index+1 == sent_ner_out.entities[i+1].start_char_index:
                    entity = Entity(
                        text = entity.text + ' ' + sent_ner_out.entities[i+1].text,
                        type = entity.type,
                        start_char_index = entity.start_char_index,
                        end_char_index = sent_ner_out.entities[i+1].end_char_index
                    )
                    i += 1
                i += 1
                new_entities.append(entity)
            sent_ner_out.entities = new_entities

    @classmethod
    def remove_punctuations_at_head_end(cls, ner_outs):
        for sent_ner_out in ner_outs:
            new_entities = []
            for entity in sent_ner_out.entities:
                if entity.text[-1] != ')' and punctuations_at_end.search(entity.text):
                    entity = cls._remove_chars_at_end(entity, offset=1)
                elif entity.text[0] == '"':
                    entity = cls._remove_chars_at_head(entity, offset=1)
                new_entities.append(entity)
            sent_ner_out.entities = new_entities

    @classmethod
    def _remove_chars_at_end(cls, entity:Entity, offset=1):
        left_move_offset =  0 - offset
        _entity = Entity(
                            text = entity.text[:left_move_offset],
                            type = entity.type,
                            start_char_index = entity.start_char_index,
                            end_char_index = entity.end_char_index + left_move_offset
                        )
        return _entity

    @classmethod
    def _remove_chars_at_head(cls, entity:Entity, offset=1):
        _entity = Entity(
                            text = entity.text[offset:],
                            type = entity.type,
                            start_char_index = entity.start_char_index + offset,
                            end_char_index = entity.end_char_index
                        )
        return _entity

    @classmethod
    def create_new_entity(cls, entity:Entity, right_text:str, end_word):
        end_index = right_text.index(end_word) + len(end_word)
        _entity = Entity(
                            text = entity.text + right_text[:end_index],
                            type = entity.type,
                            start_char_index = entity.start_char_index,
                            end_char_index = entity.end_char_index + end_index
                        )
        return _entity
