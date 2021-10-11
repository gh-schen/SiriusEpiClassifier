import logging
from sys import argv
from Classifier import regData
from configData import configData
import pickle

from dataInterface import read_features, load_molcounts_data, set_roc, convert_roc_map_to_dataframe, dump_prediction_result

"""
Only build model with the input full data and dump with pickle
"""

def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    config_path = argv[1]
    config_data = configData(config_path)

    pickle_items = []
    for suffix in ["predictor", "scaler", "transformer"]:
        infile = open(config_data.model_prefix + "." + suffix + ".pkl", 'rb')
        pcontent = pickle.load(infile)
        pickle_items.append(pcontent)
        infile.close()
    
    features = read_features(config_data.feature_path, config_data.bad_cohorts, config_data.bad_batches)
    logging.info("Read %d samples with features.", features.shape[0])
    mcm_data, raw_regions = load_molcounts_data(config_data.count_path, features, config_data.cancer_type, config_data.maf_key)
    logging.info("Loaded %d %s/normal data in %d regions.", mcm_data.shape[0], config_data.cancer_type, len(raw_regions))

    # manually change some reg_data params, as in building models
    reg_data = regData(config_data)
    reg_data.test_only = True

    reg_data.trained_model = pickle_items[0]
    reg_data.scale_model = pickle_items[1]
    reg_data.pca_model = pickle_items[2]
    reg_data.set_cv_data(mcm_data, raw_regions, config_data.iteration_start_seed)
    reg_data.run_predict_only()
    
    roc_result = reg_data.get_roc()
    if config_data.binary:
        r2_result = None
    else:
        r2_result = reg_data.get_r2_stats_dataframe(0.95) 
    pred_dataframe = reg_data.get_per_sample_logit_mafs()

    roc_map = {}
    set_roc(roc_map, roc_result, config_data.num_digits - 1)
    final_roc = convert_roc_map_to_dataframe(roc_map, config_data.num_digits)

    pred_dataframe.index = pred_dataframe["samples"]
    pred_dataframe.pop("samples")

    dump_prediction_result(config_data.output_prefix, final_roc, r2_result, pred_dataframe, reg_data.output_metrics)
    logging.info("Finished prediction at %s", config_data.output_prefix)


if __name__ == "__main__":
    main()
