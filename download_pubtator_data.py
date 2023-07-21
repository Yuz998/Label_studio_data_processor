import sys
import os
import json 
import argparse
import urllib.request
import multiprocessing
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import logging
logger = logging.getLogger("console-logger")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logging.basicConfig(filename='logs/log_download_data.txt', level=logging.DEBUG, filemode='w', format='【%(asctime)s】 【%(levelname)s】 >>>  %(message)s', datefmt = '%Y-%m-%d %H:%M')

# download abstact text and NER annotation in pubtator format
def download_abstract(X):
    _id_s, tar_dir, url_prefix = X
    file_path = tar_dir+_id_s
    url_s = url_prefix+_id_s
    # only retrive gene/disease
    # url_s += '&concepts=gene,disease'
    url_s += '&concepts=gene,chemical'

    FLAG = 0
    err_stat = ""
    
    try:
        txt = urllib.request.urlopen(url_s).read()
        if txt == '':
            err_stat = "loaded empty"
            FLAG = 2
        else:
            try:
                with open(file_path, 'wb') as F:
                    F.write(txt)

                FLAG = 1
                err_stat = "succeed"
            except Exception as e:
                err_stat = "abs loads or writting wrror"
                FLAG = 0
    except Exception as e:
        err_stat = "request error"
        FLAG = 0
    return (FLAG, err_stat, _id_s)


def write_json_file(f_name, data):
    with open(f_name, 'w') as f:
        json.dump(data, f)

# download full-text and NER annotation in biojson format
def download_full_text(X):
    _id_s, tar_dir, url_prefix = X
    file_path = tar_dir+_id_s
    url_s = url_prefix+_id_s
    # only retrive gene/disease
    # url_s += '&concepts=gene,disease'
    url_s += '&concepts=gene,chemical'
    FLAG = 0
    err_stat = ""
    
    try:
        url = urllib.request.urlopen(url_s)
        doc_dec = url.read().decode()
        if doc_dec == '':
            err_stat = "loaded empty"
            FLAG = 2
        else:
            try:
                data = json.loads(doc_dec)
                write_json_file(file_path, data)
                FLAG = 1
                err_stat = "succeed"
            except Exception as e:
                err_stat = "json loads or writting wrror"
                FLAG = 0
    except Exception as e:
        err_stat = "request error"
        FLAG = 0
    return (FLAG, err_stat, _id_s)

def multiprocessing_download(tar_id_lst, tar_dir, url_prefix, _type='json', cores=3):
    rst_rec = []
    try:
        pool = multiprocessing.Pool(processes=cores)
        _i = 0
        end_i = len(tar_id_lst)
        while _i < end_i:
            _n = min(end_i, _i + cores) - _i
            _input = [(tar_id_lst[i], tar_dir, url_prefix) for i in range(_i, _i+_n)]
            _i += _n
            tmp_i = 0
            tar_downloader = download_full_text if _type == 'json' else download_abstract
            for FLAG, err_stat, _id_s in tqdm(pool.imap(tar_downloader, _input), desc="download data: ", ncols=100):
                if (_i + tmp_i) % 100 == 0:
                    logger.info("{}, {}, {}".format((_i + tmp_i), url_prefix+_id_s, err_stat))
                tmp_i += 1
                
                rst_rec.append((_id_s, FLAG))
                if FLAG == 0:
                    logger.info("{}, {}".format(_id_s, err_stat))
                pass
    except Exception as e: 
        logging.error(e)
    #print(rst_rec)
    _hit_n = sum([_r[1] == 1 for _r in rst_rec])
    _miss_n = sum([_r[1] == 0 for _r in rst_rec])
    _empty_n = sum([_r[1] == 2 for _r in rst_rec])

    logger.info('download ended, hit {} | empty {} | miss {}'.format(_hit_n, _empty_n, _miss_n)) 
    hit_rec = [_r[0] for _r in rst_rec if _r[1] == 1]
    return hit_rec

def download_abstract_from_pubtator(tar_pmid_lst, tar_dir, cores = 3):
    rst_rec = []
    try:
        # download abstract on 'pubtator' format
        # www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/Format.html 
        if len(tar_pmid_lst) > 0:
            logger.info('begin requesting abs data, {}'.format(len(tar_pmid_lst)))
            # url_prefix = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids="
            url_prefix = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids="
            rst_rec = multiprocessing_download(tar_pmid_lst, tar_dir, url_prefix, 'abs', cores)
    except Exception as e: 
        logging.error(e)
    return rst_rec

def download_full_text_from_pubtator(tar_pmcid_lst, tar_dir, cores = 3):
    rst_rec = []
    try:
        if len(tar_pmcid_lst) > 0:
            logger.info('begin requesting full-text data, {}'.format(len(tar_pmcid_lst)))
            url_prefix = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson?pmcids="
            rst_rec = multiprocessing_download(tar_pmcid_lst, tar_dir, url_prefix, 'json', cores)
    except Exception as e: 
        logging.error(e)
    return rst_rec


def download_data(config):

    id_f = config.id_f
    is_ft = True if config.type == "ft" else False
    tar_dir = config.dir
    tmp_f = config.tmp_hit_f
    if tar_dir[-1] != '/':
        tar_dir = tar_dir + '/'

    start_time = datetime.now()

    cores = multiprocessing.cpu_count()
    cores = min(config.process_n, cores)

    try:
        id_df = pd.read_csv(id_f, header=0, dtype=str)
        id_df = id_df.drop_duplicates()

        if is_ft:
            tar_pmcid_lst = list(id_df.pmcid.values)
            rst_rec = download_full_text_from_pubtator(tar_pmcid_lst, tar_dir, cores)
        else:
            tar_pmid_lst = list(id_df.pmid.values)
            rst_rec = download_abstract_from_pubtator(tar_pmid_lst, tar_dir, cores)

        logger.info('hit records at {}'.format(tmp_f))
        pd.DataFrame(rst_rec, columns=['pmcid' if is_ft else 'pmid']).to_csv(tmp_f, index=False)

    except Exception as e: 
        logging.error(e)
    
    end_time = datetime.now()
    logger.info("**********Use time: {}".format(end_time - start_time))


def main():
    parser = argparse.ArgumentParser(description="download abstrcts/full-text pubtator/json file from Pubtator")

    parser.add_argument('--id_f', type=str, default="../data/test_download_pmid_list.csv",
                        help="PMID/PMCID list file input, default: %(default)s")

    parser.add_argument('--type', type=str, default="abs",
            help="[abs, ft] download text type: abstrcts or full-text default: %(default)s")

    parser.add_argument('--dir', type=str, default="../data/raw_data/abs/",
            help="output dir default: %(default)s")

    parser.add_argument('--tmp_hit_f', type=str, default="../data/hit_id_l.csv",
            help="output hit id list f default: %(default)s")

    parser.add_argument('--process_n', type=int, default=3,
                        help="cores number of multoprocessing")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    download_data(args)


if __name__ == "__main__":
    main()
