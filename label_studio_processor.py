import os
import warnings
from entity_linker.faiss_linker import *
from entity_linker.entity_filter import *
from entity_linker.utils_ner_basic import *
from utils.util_data import *

import json
import spacy
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
from label_studio_sdk import Client
from scispacy.abbreviation import AbbreviationDetector
pandarallel.initialize(nb_workers=50, use_memory_fs=False, progress_bar=True)

warnings.filterwarnings("ignore")

NLP_client = spacy.load("en_core_sci_lg")
NLP_client.add_pipe("abbreviation_detector")

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from datetime import datetime 
date = datetime.now().strftime("%Y-%m-%d")

class CTD2LabelStudioProcessor(LabelStudioDownloader):
    def __init__(self, api_key, label_studio_url, 
                project_name, project_id, 
                view_id, data_dir, cache_dir,
                article_abbr_path,
                device, use_gpu, have_gene,
                have_chemical,
                have_herb,
                have_ingredient,
                do_linking):
        super().__init__(api_key, label_studio_url, 
                         project_name, project_id, 
                         view_id, data_dir)
        
        self.api_key = api_key
        self.label_studio_url = label_studio_url
        self.project_name = project_name
        self.project_id = project_id
        self.view_id = view_id
        self.data_dir = os.path.join(data_dir, "data")
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu
        self.have_gene = have_gene
        self.have_chemical = have_chemical
        self.have_herb = have_herb
        self.have_ingredient = have_ingredient
        self.do_linking = do_linking
        self.output_dir = os.path.join(data_dir, f"output_{date}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.ADor = SpacyAB3P2AbbreviationDetector()
        if have_gene:
            self.FELer_gene = Faiss2EntityLinker(
                self.cache_dir, "Gene", self.use_gpu, device)
            self.filter_gene = EntityFilter(["gene"], "dbid")
        if have_chemical:
            self.FELer_chem = Faiss2EntityLinker(
                self.cache_dir, "Chemical", self.use_gpu, device)
            self.filter_chem = EntityFilter(["chemical"], "dbid")
        if have_herb:
            self.FELer_herb = Faiss2EntityLinker(
                self.cache_dir, "Herb", self.use_gpu, device)
            self.filter_herb = EntityFilter(["herb"], "dbid")
        if have_ingredient:
            self.FELer_ingredient = Faiss2EntityLinker(
                self.cache_dir, "Ingredient", self.use_gpu, device)
            self.filter_ingredient = EntityFilter(["ingredient"], "dbid")
        self.df_article_abbr = pd.read_json(article_abbr_path)

    def match_abbreviation_entity(self, guid, sent, abbr, dict_ent, lst_ent_text, anned_ent_text, ls_ent_id, 
                                  Entity_orig, lst_rel_infos, dict_article_abbr):
        new_ent, new_rel, dict_new_meta = None, None, None
        if anned_ent_text not in lst_ent_text:
            Entity_orig.text = anned_ent_text
            if self.do_linking:
                if pd.isna(dict_article_abbr) and dict_article_abbr and (
                    anned_ent_text in list(dict_article_abbr.values())):
                    new_ent_name = get_keys(dict_article_abbr, anned_ent_text)
                    tmp_ent_name = eval(new_ent_name[0])[1] if new_ent_name else anned_ent_text
                    Entity_orig.text = tmp_ent_name
                dict_new_meta = self.entity_linking(
                                    Entity_orig, sent, guid)
            new_ent = create_entity_dict(
                abbr._.long_form.start_char, abbr._.long_form.end_char, 
                anned_ent_text, dict_ent.get('value').get('labels'), dict_new_meta)
            for dict_rel in lst_rel_infos:
                if ls_ent_id == dict_rel["from_id"]:
                    new_rel = create_relation_dict(
                        new_ent["id"], dict_rel["labels"], dict_rel["to_id"], dict_rel["direction"])
                if ls_ent_id == dict_rel["to_id"]:
                    new_rel = create_relation_dict(
                        dict_rel["from_id"], dict_rel["labels"], new_ent["id"], dict_rel["direction"])
        return new_ent, new_rel

    def entity_linking(self, entity_orig, sent, guid):
        anned_ent_type, entity_name = entity_orig.type, entity_orig.text
        ent_ini, ent_end = entity_orig.start_char_index, entity_orig.end_char_index
        dict_ent_linked = {"entity_name": entity_name,
                           "entity_span": f"{ent_ini}#{ent_end}",
                           "linked_entity_name": "",
                           "entity_ID": "",
                           "entity_type": anned_ent_type,
                           "top_distance": "", 
                           "is_safe": "0"}
        entity_infos = None
        if anned_ent_type.lower() in ["gene", "ggp"]:
            ELinker = self.FELer_gene
            EFilter = self.filter_gene
        elif anned_ent_type.lower() in ["chemical"]:
            ELinker = self.FELer_chem
            EFilter = self.filter_chem
        elif anned_ent_type.lower() in ["herb"]:
            ELinker = self.FELer_herb
            EFilter = self.filter_herb
        elif anned_ent_type.lower() in ["ingredient"]:
            ELinker = self.FELer_ingredient
            EFilter = self.filter_ingredient

        linked_name, distance, extra_info = ELinker.search_vector(
            [entity_name])[0]
        linked_entity = [LinkedEntity(
            entity_orig, linked_name, distance, extra_info)]
        sent_ent_infos = SentenceNerOutput(guid, sent, linked_entity)
        entity_infos = EFilter.filter_entity(sent_ent_infos)
            
        if entity_infos is not None:
            res_entity_infos = []
            for entity_info in entity_infos:
                dict_ent_linked["entity_name"] = entity_info[3]
                dict_ent_linked["entity_span"] = f"{entity_info[1]}#{entity_info[2]}"
                dict_ent_linked["linked_entity_name"] = entity_info[4]
                dict_ent_linked["entity_ID"] = entity_info[5]
                dict_ent_linked["entity_type"] = entity_info[0]
                dict_ent_linked["top_distance"] = str(entity_info[6])
                dict_ent_linked["is_safe"] = str(entity_info[7])
                res_entity_infos.append(dict_ent_linked)
            entity_infos = res_entity_infos
        return entity_infos
    
    def check_annotation(self, x):
        x = x.copy()
        anns = x.annotations.tolist()[0][0]["result"]
        sent = x.data.tolist()[0]["text"]
        guid = x.sid.tolist()[0]
        dict_article_abbr = x.dict_abbr.tolist()[0]
        flag_abbr = 0 if pd.isna(dict_article_abbr) and dict_article_abbr else 1
        lst_ent_infos = []
        for ann in anns:
            if ann.get('from_name') == 'label':
                if ann['value']["text"].endswith(" protein"):
                    ann['value']["text"] = ann['value']["text"].strip(
                        " protein")
                    ann['value']["end"] = ann['value']["end"] - len(" protein")
                elif ann['value']["text"].endswith(" proteins"):
                    ann['value']["text"] = ann['value']["text"].strip(
                        " proteins")
                    ann['value']["end"] = ann['value']["end"] - \
                        len(" proteins")
                lst_ent_infos.append(ann)

        lst_rel_infos = [ann for ann in anns if ann.get('type') == 'relation']
        dict_abbrs = self.ADor.extract_abbreviation_spacy_ab3p(NLP_client, sent)
        lst_ent_text = [ann.get('value').get('text') for ann in lst_ent_infos]
        lst_new_ents = []
        lst_new_rels = []

        for dict_ent in lst_ent_infos:
            ls_ent_id = dict_ent.get('id')  # labelstudio 中标注的entity id
            #anned_ent_end, anned_ent_text, anned_ent_ini, anned_ent_type = dict_ent.get('value').values().tolist()
            anned_ent_text = dict_ent.get('value').get(
                'text')  # labelstudio 中标注的entity
            anned_ent_type = dict_ent.get('value').get(
                'labels')[0]  # labelstudio 中标注的entity id
            anned_ent_ini = int(dict_ent.get('value').get(
                'start'))  # labelstudio 中标注的entity span ini
            anned_ent_end = int(dict_ent.get('value').get(
                'end'))  # labelstudio 中标注的entity span end
            Entity_orig = Entity(
                anned_ent_text, anned_ent_type, anned_ent_ini, anned_ent_end)
            dict_meta = None
            if self.do_linking:
                if flag_abbr and (anned_ent_text in dict_article_abbr.values()):
                    new_ent_name = get_keys(dict_article_abbr, anned_ent_text)
                    tmp_ent_name = eval(new_ent_name[0])[1] if new_ent_name else anned_ent_text
                    Entity_orig.text = tmp_ent_name
                dict_meta = self.entity_linking(Entity_orig, sent, guid)
            dict_ent["meta"] = dict_meta
            
            # 找出对应的long form/short form
            for long_short_pair, abbr in dict_abbrs.items():
                # match short form then check if long form is in the entity list
                # import pdb;pdb.set_trace()
                if anned_ent_text == long_short_pair[0]:
                    new_ent, new_rel = self.match_abbreviation_entity(guid, sent, abbr, dict_ent, lst_ent_text, 
                                                   long_short_pair[1], ls_ent_id, Entity_orig, lst_rel_infos, dict_article_abbr)
                    if new_ent is not None:
                        lst_new_ents.append(new_ent)
                    if new_rel is not None:
                        lst_new_rels.append(new_rel)
                # match long form then check if short form is in the entity list
                if anned_ent_text == long_short_pair[1]:
                    new_ent, new_rel = self.match_abbreviation_entity(guid, sent, abbr, dict_ent, lst_ent_text, 
                                                   long_short_pair[0], ls_ent_id, Entity_orig, lst_rel_infos, dict_article_abbr)
                    if new_ent is not None:
                        lst_new_ents.append(new_ent)
                    if new_rel is not None:
                        lst_new_rels.append(new_rel)

        if lst_new_ents:
            lst_ent_infos = lst_new_ents + lst_ent_infos
        if lst_new_rels:
            lst_rel_infos += lst_new_rels
        return pd.Series([sent, lst_ent_infos, lst_rel_infos], index=["text", "ner_info", "re_info"])

    def process_annotated_json(self):
        # self.fname = "project-6-at-2023-05-10-01-47-410fbde3.json"
        df_ctd_data = pd.read_json(
            f'{self.data_dir}/{self.fname}')
        # df_ctd_data = df_ctd_data[df_ctd_data["id"]
        #                           <= 21750].reset_index(drop=True)
        # df_ctd_data = df_ctd_data.iloc[:5]
        df_ctd_data["sid"] = df_ctd_data["id"]
        df_ctd_data = df_ctd_data.merge(self.df_article_abbr[["id", "dict_abbr"]], how="left", on=["id"])
        tqdm.pandas(desc="updating annotation")
        # self.df_checked_anned = df_ctd_data.groupby(
        #     by=["id"]).progress_apply(lambda x: self.check_annotation(x)).reset_index(drop=False)
        self.df_checked_anned = df_ctd_data.groupby(
            by=["id"]).parallel_apply(lambda x: self.check_annotation(x)).reset_index(drop=False)
        tqdm.pandas(desc="creating label studio data")
        self.df_checked_anned["label_studio_data"] = self.df_checked_anned.progress_apply(
            lambda x: create_ls_json(x), axis=1)

        # project_name = "ctd_anned_yicong_v3"
        with open(f"{self.output_dir}/{self.project_name}.json", "w", encoding="utf-8") as wf:
            json.dump(self.df_checked_anned.label_studio_data.tolist(), wf,
                      ensure_ascii=False, indent=4, separators=(',', ':'))
        df_ner_re = self.df_checked_anned.apply(
            lambda x: extract_ner_infos(x), axis=1)
        
        df_ner = df_ner_re[df_ner_re["ner_infos"] != ""][["ner_infos"]]
        df_re = df_ner_re[df_ner_re["re_infos"] != ""][["re_infos"]]
        df_ner["ner_info"] = df_ner["ner_infos"].apply(lambda x: x.split("&&"))
        df_ner = df_ner.explode(column=["ner_info"])
        df_ner = df_ner["ner_info"].str.split('\t', expand=True)
        df_ner.columns = ['entity_name', "entity_span", 'linked_entity_name', 'entity_ID', "entity_type",
                          'top_distance', 'is_safe', "eid", 'tid']
        df_ner.drop_duplicates().to_csv(
            f"{self.output_dir}/ner_infos_{date}.csv", index=False)

        df_re["re_info"] = df_re["re_infos"].apply(lambda x: x.split("&&"))
        df_re = df_re.explode(column=["re_info"])
        df_re = df_re["re_info"].str.split('\t', expand=True)
        df_re.columns = ["tid", "head_eid", "tail_eid", "relations", "direction"]
        df_re = df_re.merge(df_ner.rename(columns={'entity_name': "head_name",
                                                   'entity_span': "head_span",
                                                   'linked_entity_name': "linked_head_name",
                                                   'entity_ID': "head_dbid",
                                                   'entity_type': "head_type",
                                                   'top_distance': "head_top_distance",
                                                   'is_safe': "head_is_safe",
                                                   "eid": "head_eid"}
                                          ), on=["tid", "head_eid"], how="left"
                            ).merge(df_ner.rename(columns={'entity_name': "tail_name",
                                                           'entity_span': "tail_span",
                                                           'linked_entity_name': "linked_tail_name",
                                                           'entity_ID': "tail_dbid",
                                                           "entity_type": "tail_type",
                                                           'top_distance': "tail_top_distance",
                                                           'is_safe': "tail_is_safe",
                                                           "eid": "tail_eid",
                                                           }
                                                  ), on=["tid", "tail_eid"], how="left")
        df_re["relation"] = df_re["relations"].apply(lambda x: x.split("|"))
        df_re = df_re.explode(column=["relation"])
        df_re.drop_duplicates().to_csv(
            f"{self.output_dir}/re_infos_{date}.csv", index=False)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--api_key",
        type=str,
        default="f58ab9df66f3b0ac1ef1ab953031aff1237e01c2"
    )
    
    parser.add_argument(
        "--label_studio_url",
        type=str,
        default="http://52.83.197.57:8085"
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="test"
    )

    parser.add_argument(
        "--project_id",
        type=str,
        default="6"
    )
    
    parser.add_argument(
        "--view_id",
        type=str,
        default="21"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./caches"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/yuzhao/yuzhao/data/label_studio/CTD"
    )

    parser.add_argument(
        "--article_abbr_path",
        type=str,
        default="/data/yuzhao/data/CTD/label_studio_processed/output/ctd_anned_abbr.json"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="2"
    )

    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--have_gene", action="store_true", default=False)
    parser.add_argument("--have_chemical", action="store_true", default=False)
    parser.add_argument("--have_herb", action="store_true", default=False)
    parser.add_argument("--have_ingredient", action="store_true", default=False)
    parser.add_argument("--do_linking", action="store_true", default=False)
    args = parser.parse_args()

    # Define the URL where Label Studio is accessible and the API key for your user account
    # project_name = "CTD_pranned_yicong_ocessed"
    # project_id = 6
    # view_id = 21
    # use_gpu = 0
    # have_gene = 0
    # have_chemical = 0
    # have_herb = 0
    # have_ingredient = 0
    # do_linking = 1
    # cache_dir = "./caches"
    # data_dir = "/home/yuzhao/yuzhao/data/label_studio/CTD"
    # label_studio_url = 'http://52.83.197.57:8085'
    # api_key = 'f58ab9df66f3b0ac1ef1ab953031aff1237e01c2'
    # LSDer = CTD2LabelStudioProcessor(api_key, label_studio_url,
    #                                 project_name, project_id, view_id, 
    #                                 data_dir, cache_dir, use_gpu, 
    #                                 have_gene, have_chemical, have_herb, 
    #                                 have_ingredient,  do_linking)
    LSDer = CTD2LabelStudioProcessor(args.api_key, args.label_studio_url,
                                    args.project_name, args.project_id, args.view_id, 
                                    args.data_dir, args.cache_dir, args.article_abbr_path, args.device, args.use_gpu,
                                    args.have_gene, args.have_chemical, args.have_herb, 
                                    args.have_ingredient, args.do_linking)
    # LSDer.download_label_studio_data()
    LSDer.process_annotated_json()
    LSDer.df_checked_anned.copy().reset_index(drop=False)


if __name__ == "__main__":
    main()

## CTD RE v1
# python label_studio_processor.py \
#     --api_key f58ab9df66f3b0ac1ef1ab953031aff1237e01c2 \
#     --label_studio_url http://52.83.197.57:8085 \
#     --data_dir /mnt/nas/yuzhao/data/ctd/label_studio_processed \
#     --article_abbr_path /mnt/nas/yuzhao/data/ctd/label_studio_processed/output/ctd_anned_abbr.json \
#     --project_name CTD_anned_yicong_processed_text \
#     --project_id 6 \
#     --view_id 21 \
#     --device 3 \
#     --have_gene \
#     --have_chemical \
#     --use_gpu \
#     --do_linking