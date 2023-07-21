import os
import random
import string
import subprocess
import pandas as pd
from label_studio_sdk import Client

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]
    

def create_entity_dict(start_char, end_char, text, labels, dict_meta=None):
    eid = ''.join(random.sample(string.ascii_lowercase, 8))
    dict_ner = {
        'id': eid,
        "meta": dict_meta,
        'type': 'labels',
        'value': {'end': end_char,
                  'text': text,
                  'start': start_char,
                  'labels': labels},
        'origin': 'manual',
        'to_name': 'text',
        'from_name': 'label'
    }
    if dict_meta is None:
        dict_ner.pop("meta")
    return dict_ner


def create_relation_dict(head_id, labels, tail_id, direction):
    return {
        'type': 'relation',
        'to_id': tail_id,
        'labels': labels,
        'from_id': head_id,
        'direction': direction
    }


class SpacyAB3P2AbbreviationDetector():
    def __init__(self) -> None:
        pass

    def extract_abbreviation_spacy(self, NLP_client, text):
        self.spacy_abbr_dict = {}
        doc = NLP_client(text)
        for abbr in doc._.abbreviations:
            self.spacy_abbr_dict[(abbr.text, abbr._.long_form.text)] = abbr
        return self.ab3p_abbr_dict

    def extract_abbreviation_ab3p(self, text):
        self.ab3p_abbr_dict = {}
        abbr_info = subprocess.run(
            ["python", "abbreviation_extractor.py", "--text", str(text)], capture_output=True, text=True).stdout
        if abbr_info != "":
            lines = abbr_info.split("\n")[:-1]
            for line in lines:
                full_name, short_name, _ = line.split("|")
                self.ab3p_abbr_dict[(short_name, full_name)] = short_name
        return self.ab3p_abbr_dict

    def extract_abbreviation_spacy_ab3p(self, NLP_client, text):
        self.abbr_dict = {}
        self.extract_abbreviation_ab3p(text)
        self.extract_abbreviation_spacy(NLP_client, text)
        for abbr_key, abbr in self.spacy_abbr_dict:
            if abbr_key in self.ab3p_abbr_dict:
                self.abbr_dict[abbr_key] = abbr
        return self.abbr_dict


def extract_abbreviation(NLP_client, text, do_scispacy=False):
    abbr_info = subprocess.run(
        ["python", "abbreviation_extractor.py", "--text", str(text)], capture_output=True, text=True).stdout
    ab3p_abbr_dict = {}
    if abbr_info != "":
        lines = abbr_info.split("\n")[:-1]
        for line in lines:
            full_name, short_name, precision = line.split("|")
            ab3p_abbr_dict[(short_name, full_name)] = short_name
    if do_scispacy:
        return ab3p_abbr_dict
    doc = NLP_client(text)
    abbr_dict = {}
    for abbr in doc._.abbreviations:
        abbr_key = (abbr.text, abbr._.long_form.text)
        if abbr_key in ab3p_abbr_dict:
            abbr_dict[abbr_key] = abbr
    return abbr_dict



def create_ls_json(x):
    dict_data = {"sid": x.id, "text": x.text}
    results = x.ner_info + x.re_info
    label_studio_data = {
        "data": dict_data,
        "annotations": [{"result": results}]
    }
    return label_studio_data


def extract_ner_infos(x):
    ner_infos, re_infos = [], []
    ann_res = x.label_studio_data["annotations"][0]["result"]
    for ner_re_info in ann_res:
        if ner_re_info['type'] == "labels":
            if (len(ner_re_info["meta"]) != 0):
                for d_ner in ner_re_info["meta"]:
                    d_ner["eid"] = ner_re_info["id"]
                    d_ner["tid"] = x.id
                    ner_infos.append("\t".join(str(i)
                                               for i in list(d_ner.values())))
            else:
                d_ner = {'entity_name': ner_re_info["value"]["text"],
                         "entity_span": "",
                         'linked_entity_name': "",
                         'entity_ID': "",
                         "entity_type": ner_re_info["value"]['labels'][0],
                         'top_distance': "",
                         'is_safe': '0',
                         "eid": ner_re_info["id"],
                         'tid': x.id}
                ner_infos.append("\t".join(str(i)
                                    for i in list(d_ner.values())))
            # import pdb;pdb.set_trace()
        if ner_re_info['type'] == "relation":
            head_id, tail_id = ner_re_info["from_id"], ner_re_info["to_id"]
            relations = "|".join(ner_re_info["labels"])
            direction = ner_re_info["direction"]
            re_infos.append("\t".join(str(i) for i in [
                            str(x.id), head_id, tail_id, relations, direction]))

    return pd.Series(["&&".join(ner_infos), "&&".join(re_infos)], index=["ner_infos", "re_infos"])

class LabelStudioDownloader:
    def __init__(self,
                 api_key,
                 label_studio_url,
                 project_name,
                 project_id,
                 view_id,
                 data_dir,
                 ):
        
        self.api_key = api_key
        self.label_studio_url = label_studio_url
        self.project_name = project_name
        self.project_id = project_id
        self.view_id = view_id
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.fname = f"{self.project_name}_p{self.project_id}_v{self.view_id}.json"

    def download_label_studio_data(self):
        # Connect to the Label Studio API and check the connection
        ls_client = Client(url=self.label_studio_url, api_key=self.api_key)
        ls_client.check_connection()
        project = ls_client.get_project(self.project_id)  # project object
        # create new export snapshot
        export_result = project.export_snapshot_create(
            title=self.project_name,
            task_filter_options={"view": self.view_id}
        )
        export_id = export_result['id']
        # download snapshot file
        status, json_fname = project.export_snapshot_download(
            export_id=export_id, export_type='JSON', path=self.data_dir)
        assert status == 200
        ann_path = os.path.join(self.data_dir, json_fname)
        new_ann_path = os.path.join(self.data_dir, self.fname)
        os.rename(ann_path, new_ann_path)
        logger.info(f"Downloaded {self.project_name} .")