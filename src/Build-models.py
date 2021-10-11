#!/usr/bin/env python3

import logging
from sys import argv
from Classifier import regData
from configData import configData
import pickle

from dataInterface import read_features, load_molcounts_data

"""
Only build model with the input full data and dump with pickle
"""

def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    config_path = argv[1]
    config_data = configData(config_path)

    features = read_features(config_data.feature_path, config_data.bad_cohorts, config_data.bad_batches)
    logging.info("Read %d samples with features.", features.shape[0])
    mcm_data, raw_regions = load_molcounts_data(config_data.count_path, features, config_data.cancer_type, config_data.maf_key)
    logging.info("Loaded %d %s/normal data in %d regions.", mcm_data.shape[0], config_data.cancer_type, len(raw_regions))

    # manually change some reg_data params
    reg_data = regData(config_data)
    reg_data.training_only = True
    reg_data.set_cv_data(mcm_data, raw_regions, config_data.iteration_start_seed)
    logging.info("Set %d training data completed.", mcm_data.shape[0])

    reg_data.run_training()
    logging.info("Training completed.")

    outpath = config_data.output_prefix + ".predictor.pkl"
    outfile = open(outpath, 'wb')
    pickle.dump(reg_data.trained_model, outfile)
    outfile.close()

    outpath = config_data.output_prefix + ".scaler.pkl"
    outfile = open(outpath, 'wb')
    pickle.dump(reg_data.scale_model, outfile)
    outfile.close()

    outpath = config_data.output_prefix + ".transformer.pkl"
    outfile = open(outpath, 'wb')
    pickle.dump(reg_data.pca_model, outfile)
    outfile.close()

    # for diagnostic
    outpath = config_data.output_prefix + ".training_roc.tsv"
    reg_data.get_roc(rtype="train").to_csv(outpath, sep='\t', index=False)


if __name__ == "__main__":
    main()
