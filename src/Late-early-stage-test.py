#!/usr/bin/env python3

import logging
import json
from sys import argv
from scipy.sparse.construct import rand
from configData import configData
from Classifier import regData
from dataInterface import read_features, load_molcounts_data, set_roc, convert_roc_map_to_dataframe

"""
Test two datasets: late-stage only model & late+early, with fixed size
"""

def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    config_path = argv[1]
    config_data = configData(config_path)
    cancer_type = "crc"
    num_digits = 4

    features = read_features(config_data.feature_path)
    logging.info("Read %d samples with features.", features.shape[0])
    mcm_data, raw_regions = load_molcounts_data(config_data.count_path, features, cancer_type)
    logging.info("Loaded %d %s/normal data in %d regions.", mcm_data.shape[0], cancer_type, len(raw_regions))

    # generate late-stage only training set
    late_crcs = mcm_data[(mcm_data["cancer_type"]=="crc") & ((mcm_data["cohort"]=="G360") | (mcm_data["stage"]=="stage_iv"))]
    early_crcs = mcm_data[(mcm_data["cancer_type"]=="crc") & (mcm_data["cohort"]!="G360") & (mcm_data["stage"]!="stage_iv")]
    normals = mcm_data[mcm_data["cancer_type"]=="cancer_free"]
    logging.info("late early normal count = %d, %d, %d", late_crcs.shape[0], early_crcs.shape[0], normals.shape[0])

    # generate late + early set (same size, 50 50)
    sample_seed = 100
    num_late = int(late_crcs.shape[0] / 2)
    num_normal = int(normals.shape[0] / 2)
    train_late = late_crcs.sample(n=num_late, random_state=sample_seed)
    train_early = early_crcs.sample(n=num_late, random_state=sample_seed)
    train_normal = normals.sample(n=num_normal, random_state=sample_seed)
    logging.info("train early late normal: %d, %d, %d", train_early.shape[0], train_late.shape[0], train_normal.shape[0])
    test_early = early_crcs[~early_crcs.index.isin(train_early.index)]
    test_normal = normals[~normals.index.isin(train_normal.index)]

    dt_train_late = late_crcs.append(train_normal)
    dt_train_mix = train_late.append(train_early).append(train_normal)
    dt_test = test_early.append(test_normal)

    raw_prefix = config_data.output_prefix
    config_data.output_prefix = raw_prefix + ".late"
    run_iterated_testing(config_data, num_digits, dt_train_late, dt_test, raw_regions)
    config_data.output_prefix = raw_prefix + ".mix"
    run_iterated_testing(config_data, num_digits, dt_train_mix, dt_test, raw_regions)


def run_iterated_testing(config_data, num_digits, dt_train, dt_test, raw_regions):
    logging.info("Start CV.")
    roc_map = {}
    final_r2 = None
    final_pred = None
    final_metrics = []
    for cv_idx in range(config_data.total_iterations):
        shuffle_seed = config_data.iteration_start_seed + cv_idx
        r2_result, roc_result, pred_dataframe, out_metrics = run_single_iteration(dt_train, dt_test, raw_regions, config_data, shuffle_seed)
        set_roc(roc_map, roc_result, num_digits=3)
        if cv_idx == 0:
            final_r2 = r2_result
            final_pred = pred_dataframe
            final_pred.columns = ["samples", "true", "pred0", "train0", "status"]
            final_pred.index = final_pred["samples"]
            final_pred.pop("samples")
        else:
            if not config_data.binary:
                final_r2 = final_r2.append(r2_result)
            pred_dataframe.index = pred_dataframe["samples"]
            for k in ["samples", "true", "status"]:
                pred_dataframe.pop(k)
            pred_dataframe.columns = ["pred" + str(cv_idx), "train" + str(cv_idx)]
            final_pred = final_pred.merge(pred_dataframe, how='outer', left_index=True, right_index=True)
        final_metrics.append(out_metrics)

    final_roc = convert_roc_map_to_dataframe(roc_map, num_digits)
    final_roc.to_csv(config_data.output_prefix + ".roc.tsv", sep='\t', index=False)
    if not config_data.binary:
        final_r2.to_csv(config_data.output_prefix + ".r2.tsv", sep='\t', index=True)
    final_pred = final_pred.round(num_digits)
    final_pred.to_csv(config_data.output_prefix + ".pred.tsv", sep='\t', index=True)
    outfile = open(config_data.output_prefix + ".metrics.json", 'w')
    json.dump(final_metrics, outfile)
    outfile.close()


def run_single_iteration(dt_train, dt_test, input_regions, config_data, shuffle_seed):
    train_reg = regData(config_data)
    train_reg.training_only = True
    train_reg.set_cv_data(dt_train, input_regions, shuffle_seed)
    train_reg.run_training()

    test_reg = regData(config_data)
    test_reg.test_only = True
    test_reg.trained_model = train_reg.trained_model
    test_reg.scale_model = train_reg.scale_model
    test_reg.pca_model = train_reg.pca_model
    test_reg.set_cv_data(dt_test, input_regions, shuffle_seed)
    test_reg.run_predict_only()
    
    roc_result = test_reg.get_roc()
    if config_data.binary:
        r2_result = None
    else:
        r2_result = test_reg.get_r2_stats_dataframe(0.95)
    pred_dataframe = test_reg.get_per_sample_logit_mafs()
    return r2_result, roc_result, pred_dataframe, test_reg.output_metrics


if __name__ == "__main__":
    main()
