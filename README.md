# Epigenomic classifier in Sirius
Perform Tumor/normal classification and tumor fraction prediction for multiple cancers

# Usage
`python3 <script> <config_path>`

# Output
- *pred.tsv: predictions score for each sample
- *roc.tsv: ROC curve - contains cutoff
- *r2.tsv: R2 for tumor fractions (MAF as truth)

## Configs
A config file is required to run these scripts. Example config:

- Build CRC quantitative model (MAF as input):
    /ghdevhome/home/schen/epigen/ccbi-327/configs/crc_quant_modeling-config.json

- Use CRC model to predict lung data:
    /ghdevhome/home/schen/epigen/ccbi-327/configs/mc_config/lung-crc-binary.config


# List of scripts
- `Run_mcm_models.py`

    Use N-fold CV to test model performance on given sets of data.

- `Build-models.py`

    Build prediction models only. Models will be stored in .pickle

- `Run-prediction.py`

    Run prediction given .pickle models from `Build-models.py`

- `Run-TCGA-baseline.py`

    Run TCGA baseline model for specific cancer types.

- `Late-early-stage-test.py`

    Script used to generate late & early comparison: https://docs.google.com/presentation/d/1LLf-xLDdK_aC3a0jmcntjuTlE5TlnpqiyQAv7IxwYtI/edit?usp=sharing
