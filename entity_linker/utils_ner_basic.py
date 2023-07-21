""" NER specific tools """

import os
from dataclasses import dataclass
from typing import List, Optional, Union

import re
import torch
from torch.utils.data.dataset import TensorDataset
import random
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

punctuation_incl_pat = re.compile(r'([\W_])')
words_pat = re.compile(r'[^\W_]+')
sent_end_punctuations = ['?', '.', ';', '!']
digits = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


@dataclass
class InputFeaturesBert:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    input_mask: List[int]
    segment_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    input_len: Optional[int] = 0


@dataclass
class Word:
    """
    """
    text: str
    start: int = 0
    end: int = 0


@dataclass
class WordAndType:
    """
    """
    word: Word
    ner_type: Optional[str] = None


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class TestSentenceInput:
    """
    """
    input_ids: List[int]
    input_masks: List[int]
    head_masks: List[int]
    input_len: int


@dataclass(frozen=False)
class Entity:
    """ frozen = True is for duplicate because 1 sentence overlap when splitting long sentences.
    Stimulates the stanza entity output object. Entity is a dataclass with contents as below:
            {
                "text": "HCC",
                "type": "DISEASE",
                "start_char_index": 217,
                "end_char_index": 220
            }
    """
    text: str
    type: str
    start_char_index: int = 0
    end_char_index: int = 0


@dataclass
class LinkedEntity:
    """ Nice to put Entity inside, very good design!
    """
    entity: Entity
    linked_name: str
    distance: float
    extra_info: dict


@dataclass
class SentenceNerOutput:
    """
    A NER out class based on the sentence level.
    entity in entities could be Entity or LinkedEntity and so not clearly claimed
    """
    guid: str
    sentence_text: str
    entities: List[LinkedEntity]

    def __str__(self) -> str:
        entities_str = '\n'.join([str(entity) for entity in self.entities])
        out_str = f'{self.sentence_text}\nguid: {self.guid}, len(entities): {len(self.entities)}\n{entities_str}\n'
        return out_str


@dataclass
class SplitTokens:
    """ tokens from sentence_text.
    If the orig tokens length exceeds the max seq length, the orig tokens will be split as several tokens groups, and
    only the first split tokens group has orig sentence text, and the others have sentence text as None
    """
    tokens: List
    sentence_text: Optional[str] = None


def create_tensor_dataset_from_features(features:List[InputFeaturesBert]):
    """ Convert to Tensors and build dataset """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_lens)
    return dataset


def read_examples_from_file(
    data_dir,
    mode: str,
    label_postfix: str
) -> List[InputExample]:
    file_path = get_file_path(data_dir, mode)
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split()
                words.append(splits[0])
                if len(splits) > 1:
                    splits_replace = splits[-1]
                    if splits_replace == 'O':
                        labels.append(splits_replace)
                    else:
                        labels.append(f'{splits_replace}-{label_postfix}')
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
    return examples


def read_examples_from_sentences(
    sentences: List[str]
) -> List[InputExample]:
    """ No real label from sentences
    """
    guid_index = 1
    examples = []
    for sentence in sentences:
        words = []
        labels = []
        orig_words = sentence.split()
        for word in orig_words:
            # I missed to test continuously punctuations which cause some empty string in middle. Now, I fixed the bug!
            subwords = punctuation_incl_pat.split(word)
            for _word in subwords:
                if _word:
                    words.append(_word)
                    labels.append('O')
        if words:
            examples.append(InputExample(guid=f"{guid_index}", words=words, labels=labels))
            guid_index += 1
    return examples


def  get_file_path(data_dir, mode):
    file_path = os.path.join(data_dir, f"{mode}.txt")
    if not os.path.exists(file_path):
        file_path = os.path.join(data_dir, f"{mode}.tsv")
    if not os.path.exists(file_path):
        file_path = os.path.join(data_dir, f"{mode}.ck")
    if not os.path.exists(file_path):
        file_path = os.path.join(data_dir, f"{mode}.char")
    return file_path


def get_labels(corpus_type: Union[str, List[str]], add_x_at_head=False) -> List[str]:
    """
    add_x_at_head is used for CRF, labels.insert(0, 'X')
    """
    if isinstance(corpus_type, str):
        labels = ["O", f"B-{corpus_type}", f"I-{corpus_type}"]
    elif corpus_type and isinstance(corpus_type, List):
        labels = ['O']
        for type in corpus_type:
            labels.append(f"B-{type}")
            labels.append(f"I-{type}")
    else:
        raise Exception('Invalid to get labels')
    if add_x_at_head:
        labels.insert(0, 'X')
    return labels


def get_predicted_positive_num(precision, recall, support):
    """ support is the number of all the postives in true labels """
    true_postive_num = support * recall
    false_positve_num = true_postive_num * (1 - precision) / precision
    predicted_postive_num = round(true_postive_num + false_positve_num)
    return predicted_postive_num, round(true_postive_num), round(false_positve_num)


def get_all_entities_from2d_list(lst:List[List[str]]):
    """ ignore 'O' and get all the targets, only BIO
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    """
    all_entities = []
    words = []
    for sub_list in lst:
        for item in sub_list:
            if item.startswith('B'):
                if words:
                    entity = ' '.join(words)
                    all_entities.append(entity)
                    words.clear()
                words.append(item.split('-')[1])
            elif item.startswith('I') and words:
                words.append(item.split('-')[1])
    if words:
        entity = ' '.join(words)
        all_entities.append(entity)
    return all_entities


def is_sentence_end(token, token_index, tokens):
    """
    example:
        .Fixing.. 5.0% 5,000 Number is 5. Ok. 5k
    1) if token['text'] in comma_and_dot:
    1.1) If the comma_and_dot at the end of sentence, it is sentence end;
    1.2) ELIf there are both digits at head and tail of comma_and_dot, it is not sentence end.
    1.3) it is sentence end;
    2) Else:
    it is not sentence end.
    """
    if token['text'] in sent_end_punctuations:
        if token_index + 1 == len(tokens):
            return True
        # If there are both digits at head and tail of comma_and_dot, it is not sentence end.
        elif (tokens[token_index+1]['text'].startswith(digits)
                and token_index > 0 and tokens[token_index-1]['text'].startswith(digits)):
            return False
        else:
            return True
    return False


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception as identifier:
        pass


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_labels, all_lens = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_labels, all_lens


def collate_fn_test(batch):
    """
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    input_ids, input_masks, head_masks, input_lens = zip(*batch)
    max_len = max(input_lens)
    input_ids = torch.LongTensor(np.array(input_ids)[:, :max_len])
    input_masks = torch.LongTensor(np.array(input_masks)[:, :max_len])
    head_masks = torch.tensor(np.array(head_masks)[:, :max_len], dtype=torch.bool)
    return input_ids, input_masks, head_masks


def collate_fn_test_trt(batch, min_len=64):
    """
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    input_ids, input_masks, head_masks, input_lens = zip(*batch)
    ## fixed seq len
    # input_ids = torch.IntTensor(input_ids)
    # input_masks = torch.IntTensor(input_masks)
    # head_masks = torch.tensor(head_masks, dtype=torch.bool)
    ## dynamic seq len
    max_len = max(input_lens)
    # if max_len < min_len: max_len = min_len
    input_ids = torch.IntTensor(np.array(input_ids)[:, :max_len])
    input_masks = torch.IntTensor(np.array(input_masks)[:, :max_len])
    head_masks = np.array(head_masks, dtype=bool)[:, :max_len]
    # head_masks = torch.tensor(np.array(head_masks)[:, :max_len], dtype=torch.bool)
    return input_ids, input_masks, head_masks


def collate_fn_test_onnx(batch):
    """
    Returns a padded numpay array of sequences sorted from longest to shortest,
    """
    input_ids, input_masks, head_masks, input_lens = zip(*batch)
    max_len = max(input_lens)
    input_ids = np.array(input_ids)[:, :max_len]
    input_masks = np.array(input_masks)[:, :max_len]
    head_masks = np.array(head_masks, dtype=bool)[:, :max_len]
    return input_ids, input_masks, head_masks

    
def get_device(no_cuda:bool):
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    return device


def build_label_map(labels):
    label_map = {label: i for i, label in enumerate(labels)}
    return label_map

def build_id_to_label(label_map):
    return {v: k for k, v in label_map.items()}
