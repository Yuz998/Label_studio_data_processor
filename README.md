# Label_studio_data_processor

## 1. Install

### 1.1 Ab3P

Download [Ab3P](https://ftp.ncbi.nlm.nih.gov/pub/wilbur/Ab3P-v1.5.tar.gz)

### 1.2 Environments

### 1.3 Dependencies

```
Python==3.8.15
cuda==1.11
```

```
dask==2022.11.0
en-core-sci-lg==0.5.1
faiss-gpu==1.7.2
label-studio-sdk==0.0.23
label-studio-tools==0.0.2
numpy==1.23.4
pandarallel==1.6.3
pandas==1.5.1
scikit-learn==1.1.3
scispacy==0.5.2
spacy==3.4.4
tqdm==4.64.1
```

## run

```bash
# extract abbreviation
python extract_abbr.py

# process label studio data
python label_studio_processor.py \
    --api_key f58ab9df66f3b0ac1ef1ab953031aff1237e01c2 \
    --label_studio_url http://52.83.197.57:8085 \
    --data_dir /mnt/nas/yuzhao/data/ctd/label_studio_processed \
    --article_abbr_path /mnt/nas/yuzhao/data/ctd/label_studio_processed/output/ctd_anned_abbr.json \
    --project_name CTD_anned_yicong_processed_text \
    --project_id 6 \
    --view_id 21 \
    --device 3 \
    --have_gene \
    --have_chemical \
    --use_gpu \
    --do_linking

```