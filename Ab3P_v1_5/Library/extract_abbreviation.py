import os
import subprocess
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from pandarallel import pandarallel

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

delete_uid = ["20664721-TABLE-318", "35813599-TABLE-698", 
              "35813599-TABLE-700", "35813599-TABLE-702",
              "35813599-TABLE-704", "35813599-TABLE-707", 
              "35813599-TABLE-710", "35813599-TABLE-712",
              "35813599-TABLE-714", "35813599-TABLE-716",
              "35813599-TABLE-718", "35813599-TABLE-720",
              ]

def extract_abbreviation(uid, text):
        abbr_info = subprocess.run(
            ["python", "abbreviation_extractor.py", "--text", str(text)], capture_output=True, text=True).stdout
        uids, full_names, short_names, precisions = [], [], [], []
        if abbr_info != "":
            lines = abbr_info.split("\n")[:-1]
            for line in lines:
                abbr = line.split("|")
                uids.append(uid)
                full_names.append(abbr[0])
                short_names.append(abbr[1])
                precisions.append(abbr[2])
        # print(uid)
        return pd.Series([uids, full_names, short_names, precisions],
                          index=["uid", "full_name", "short_name", "precision"
                    ])


def extract_abbreviation_for_text(df, fname, is_parallel, n_jobs):
    df_sent = df.copy()
    if is_parallel:
        pandarallel.initialize(nb_workers=n_jobs, use_memory_fs=False, progress_bar=True)
        df_abbr = df_sent.parallel_apply(lambda x: extract_abbreviation(x["uid"], x["sentence"]), axis=1)
    else:
        tqdm.pandas(desc=f"extract abbreviation {fname}: ")
        df_abbr = df_sent.progress_apply(lambda x: extract_abbreviation(x["uid"], x["sentence"]), axis=1)
        # df_abbr = df_sent.apply(lambda x: extract_abbreviation(x["uid"], x["sentence"]), axis=1)
    df_abbr = df_abbr[df_abbr["uid"].map(lambda x: True if len(x)!=0 else False)]
    df_abbr = df_abbr.explode(["uid", "full_name", "short_name", "precision"])
    return df_abbr.copy()


class AbbreviationExtractor():
    def __init__(self, data_dir, data_type, table_format, n_jobs, is_multiprocess) -> None:
        self.data_type = data_type
        self.table_format = table_format
        self.n_jobs = n_jobs
        self.is_multiprocess = is_multiprocess
        self.is_parallel = not is_multiprocess
        self.sent_dir = os.path.join(data_dir, "sentence")
        self.abbr_dir = os.path.join(data_dir, "abbreviation")
        os.makedirs(self.abbr_dir, exist_ok=True)

    def extract_abbreviation_for_one_file(self, fname):
        if not os.path.exists(f"{self.abbr_dir}/{fname}.{self.table_format}"):
            df_sent = pd.read_csv(f"{self.sent_dir}/{fname}.tsv", sep="\t", dtype=str)
            df_sent = df_sent[~df_sent["sentence"].isna()].reset_index(drop=True)
            df_sent = df_sent[df_sent["sentence"].str.contains("[\[\]\(\)]")].reset_index(drop=True)
            if df_sent.shape[0]!=0:
                if self.data_type == "pubmed":
                    df_sent["uid"] = df_sent["uid"] + "-" + df_sent["version"] + "-" + df_sent["sentence_id"]
                elif self.data_type == "pmc":
                    df_sent["uid"] = df_sent["uid"] + "-" + df_sent["version"] + "-" + df_sent["sentence_id"]
                elif self.data_type == "ptc":
                    df_sent["uid"] = df_sent["docid"] + "-" + df_sent["section_type"] + "-" + df_sent["sentence_id"]
                    df_sent = df_sent[~df_sent["uid"].isin(delete_uid)].reset_index(drop=True)
                # import pdb;pdb.set_trace()
                # 35813599-CONCL-681
                df_abbr = extract_abbreviation_for_text(df_sent.copy(), fname, self.is_parallel, self.n_jobs)
                # tqdm.pandas(desc="extract abbreviation: ")
                # df_abbr = df_sent.progress_apply(lambda x: extract_abbreviation(x["uid"], x["sentence"]), axis=1)
                # df_abbr = df_abbr[df_abbr["uid"].map(lambda x: True if len(x)!=0 else False)]
                # df_abbr = df_abbr.explode(["uid", "full_name", "short_name", "precision"])

                if df_abbr.shape[0]!=0:
                    save_path = os.path.join(self.abbr_dir, f"{fname}.{self.table_format}")
                    if self.table_format in ["txt", "tsv"]:
                        df_abbr.to_csv(save_path, sep="\t", index=False)
                    elif self.table_format in ["csv"]:
                        df_abbr.to_csv(save_path, index=False)
                    elif self.table_format in ["json"]:
                        df_abbr.to_json(save_path, orient="records")
                    else:
                        logger.info("Error table format.")

    def multiprocess_extract_abbreviation(self):
        fnames = [i[:-4] for i in os.listdir(self.sent_dir)]
        if os.path.exists(self.abbr_dir):
            fnames = sorted(
                list(
                    set(fnames)
                    - set([i[:-4] for i in os.listdir(self.abbr_dir)])
                )
            )
        logger.info(f"# of file: {len(fnames)}")
        if self.is_multiprocess:
            logger.info(
                f"---Using multiprocess ({self.n_jobs}) extract {self.data_type} abbreviation---")
            with Pool(processes=self.n_jobs) as pool:
                pool.map(self.extract_abbreviation_for_one_file, fnames)
        else:
            for fname in tqdm(fnames, desc="extract abbreviation:"):
                self.extract_abbreviation_for_one_file(fname)
                logger.info(f"Finshed {fname}")


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default="/data/yuzhao/data/PubTatorCentral/BioCXML/test"
    )

    parser.add_argument(
        '--data_type', 
        type=str, 
        default="ptc",
        help="database type: pubmed, pmc, ptc"
    )

    parser.add_argument(
        '--table_format', 
        type=str, 
        default="csv"
    )

    parser.add_argument(
        "--n_jobs", 
        type=int, 
        default=30
    )

    # explore args, disable
    parser.add_argument("--is_multiprocess", action="store_true", default=False)
    args = parser.parse_args()

    AEor = AbbreviationExtractor(args.data_dir, args.data_type, args.table_format, args.n_jobs, args.is_multiprocess)
    AEor.multiprocess_extract_abbreviation()


if __name__ == "__main__":
    main()

# python extract_abbreviation.py \
#     --data_dir=/data/yuzhao/data/PubTatorCentral/BioCXML/test \
#     --data_type=ptc \
#     --table_format=csv \
#     --n_jobs=10 \
#     --is_multiprocess