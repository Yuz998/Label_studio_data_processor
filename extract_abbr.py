import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=50, use_memory_fs=False, progress_bar=True)

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from utils.util_data import *


class Ab3PAbbreviationExtractor():
    def __init__(self, data_dir, output_path):
        self.data_dir = data_dir
        self.output_path = output_path
        self.ADor = SpacyAB3P2AbbreviationDetector()

    def extract_sentence_abbreviation(self):
        df_pmid = pd.read_csv(f"{self.data_dir}/output/id_pmid_sentence.csv", dtype=str)
        df_abs_ti = dd.read_csv(f"{self.data_dir}/pubtator/abs/*", sep="|", names=["pmid", "article_type", "text"]).compute()
        df_tit = df_abs_ti[df_abs_ti["article_type"] == "t"].rename(columns={"text": "title"})
        df_abs = df_abs_ti[df_abs_ti["article_type"] == "a"].rename(columns={"text": "abstract"})
        df_text = pd.merge(df_tit[["pmid", "title"]], df_abs[["pmid", "abstract"]], on=["pmid"], how="outer")
        df_text["abstract"] = df_text["abstract"].fillna("")
        df_text["text"] = df_text["title"] + " " + df_text["abstract"]
        df_text["dict_abbr"] = df_text["text"].parallel_apply(lambda x: self.ADor.extract_abbreviation_ab3p(x))
        df_res = df_pmid.merge(df_text[["pmid", "text", "dict_abbr"]], on=["pmid"], how="left").drop_duplicates(subset=["id", "pmid", "text"])
        df_res.to_json(self.output_path, orient="records")


def main():
    data_dir = "/data/yuzhao/data/CTD/label_studio_processed"
    output_path = "/data/yuzhao/data/CTD/label_studio_processed/output/ctd_anned_abbr.json"
    Ab3Por = Ab3PAbbreviationExtractor(data_dir, output_path)
    Ab3Por.extract_sentence_abbreviation()


if __name__ == "__main__":
    main()

