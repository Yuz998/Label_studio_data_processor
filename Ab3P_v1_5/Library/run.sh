export LD_LIBRARY_PATH=/data/yuzhao/code/PubMedSearcher/Ab3P/Ab3P_v1/Library:$LD_LIBRARY_PAT
data_dir=/data/yuzhao/data/PubTatorCentral/BioCXML/7
data_type=ptc

python extract_abbreviation.py \
    --data_dir=${data_dir} \
    --data_type=${data_type} \
    --table_format=csv \
    --n_jobs=70 \
    --is_multiprocess
