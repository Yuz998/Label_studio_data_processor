import json
import random
import string
import pandas as pd


class LabelStudioData:
    def __init__(self, dict_metadata, text):
        self.results = []
        dict_data = {"text": text}
        dict_data.update(dict_metadata)
        self.label_studio_data = {
                "data" : dict_data,
                "annotations" : [{"result": self.results}]
            }
        
    def add_ner_result(self, text_id, text_name, text_ini, text_end, text_label:list):
        ner_result = {
                    "id": text_id,
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                            "text": text_name,
                            "start": text_ini,
                            "end": text_end,
                            "labels": text_label
                        }
                }
        self.results.append(ner_result)
    def add_re_result(self, head_id, tail_id, direction, relation_label:list):
        re_result = {
                "type" : "relation",
                "from_id" : head_id, # head
                "labels" : relation_label,
                "to_id" : tail_id, # tail
                "direction" : direction
            }
        self.results.append(re_result)

def create_ner_data(x):
    ner_infos = []
    sent = x.text.unique()[0]
    uid = x.uid.unique()[0]
    lst_dict_first = x.to_dict(orient="records")
    ner_infos.extend(lst_dict_first)
    meta_data = {"uid": uid}
    LSData = LabelStudioData(meta_data, sent)

    for ner_info in ner_infos:
        ent_ini = ner_info["new_entity_ini"]
        ent_end = ner_info["new_entity_end"]
        ent_id = ''.join(random.sample(string.ascii_lowercase, 8))
        LSData.add_ner_result(ent_id, ner_info["entity_name"], ent_ini, ent_end, [ner_info["name_type"]])
    label_studio_res = LSData.label_studio_data
    return pd.DataFrame({"uid": [uid], "NER": [ner_infos], "label_studio_data": [label_studio_res]})


def create_data(x):
    ner_infos = []
    df_first = x.drop_duplicates(subset=["first_span"])[['first_entity', 'first_type', 'first_span_new', 'first_ID']]
    df_second = x.drop_duplicates(subset=["second_span"])[['second_entity', 'second_type', 'second_span_new', 'second_ID']]
    df_first.columns = ['entity_name', 'entity_type', 'entity_span', 'entity_ID']
    df_second.columns = ['entity_name', 'entity_type', 'entity_span', 'entity_ID']
    sent = x.sentence.unique()[0]
    final_subpattern = x.final_subpattern.unique()[0]
    path_root_lemma = x.path_root_lemma.unique()[0]
    path_root_pos = x.path_root_pos.unique()[0]
    
    lst_uid = [f"{pmid}_{sent_id}" for pmid, sent_id in zip(x.pmid, x.sentence_id)]
    uid = "|".join(list(set(lst_uid)))
    lst_dict_first = df_first.to_dict(orient="records")
    lst_dict_second = df_second.to_dict(orient="records")
    ner_infos.extend(lst_dict_first)
    ner_infos.extend(lst_dict_second)
    meta_data = {"uid": uid, "final_subpattern": final_subpattern, "path_root_lemma": path_root_lemma, "path_root_pos":path_root_pos}
    LSData = LabelStudioData(meta_data, sent)

    for ner_info in ner_infos:
        ent_ini = ner_info["entity_span"].split("#")[0]
        ent_end = ner_info["entity_span"].split("#")[1]
        ent_id = ''.join(random.sample(string.ascii_lowercase, 8))
        LSData.add_ner_result(ent_id, ner_info["entity_name"], ent_ini, ent_end, [ner_info["entity_type"]])
    label_studio_res = LSData.label_studio_data
    return pd.DataFrame({"uid": [uid], "NER": [ner_infos], "label_studio_data": [label_studio_res]})


df_example = pd.read_csv(r"D:\迅雷下载\compound_x_gene_NN_path_root - Sheet1.csv", dtype=str)
dff = df_example.groupby(by="sentence").apply(lambda x: create_data(x))

with open(r"C:\Users\yuzha\Desktop\pattern_example.json", "w", encoding="utf-8") as wf:
    json.dump(dff.label_studio_data.tolist(), wf, ensure_ascii=False, indent=4, separators=(',', ':'))