#!/usr/bin/env python3

import logging
import json
from sys import argv
from typing import final
from Classifier import regData
from configData import configData
from subprocess import check_call

from dataInterface import read_features, load_molcounts_data, set_roc, convert_roc_map_to_dataframe, gen_simulated_data

"""
Gateway of running simulation & prediction & modeling
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

    if hasattr(config_data.in_silico_titration):
        simu_data = gen_simulated_data(mcm_data, raw_regions, config_data.in_silico_titration, config_data.cancer_type)

    logging.info("Start CV.")
    roc_map = {}
    final_r2 = None
    final_pred = None
    final_metrics = []
    for cv_idx in range(config_data.total_iterations):
        shuffle_seed = config_data.iteration_start_seed + cv_idx
        r2_result, roc_result, pred_dataframe, out_metrics = run_single_iteration(mcm_data, raw_regions, config_data, shuffle_seed, simu_data)
        set_roc(roc_map, roc_result, num_digits=config_data.num_digits-1)
        if cv_idx == 0:
            final_r2 = r2_result
            final_pred = pred_dataframe
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
        logging.info("Finished iteration #%d.", cv_idx)

    final_roc = convert_roc_map_to_dataframe(roc_map, config_data.num_digits)
    final_roc.to_csv(config_data.output_prefix + ".roc.tsv", sep='\t', index=False)
    if not config_data.binary:
        final_r2.to_csv(config_data.output_prefix + ".r2.tsv", sep='\t', index=True)
    final_pred = final_pred.round(config_data.num_digits)
    final_pred.to_csv(config_data.output_prefix + ".pred.tsv", sep='\t', index=True)
    outfile = open(config_data.output_prefix + ".metrics.json", 'w')
    json.dump(final_metrics, outfile)
    outfile.close()

    if not config_data.binary:
        check_call("cat " + config_data.output_prefix + ".r2.tsv", shell=True)
    cmd = "cat " + config_data.output_prefix + ".roc.tsv | awk '$1>=0.95' | head -n 2"
    check_call(cmd, shell=True)


def run_single_iteration(mcm_data, raw_regions, config_data, cv_seed, simu_data=None):
    reg_data = regData(config_data)
    reg_data.set_cv_data(mcm_data, raw_regions, cv_seed)
    if simu_data is not None:
        reg_data.set_simu_data(simu_data, raw_regions)

    logging.info("Set %d fold CV data with %d in each partition.", reg_data.num_cv_, reg_data.test_x[0].shape[0])

    reg_data.run_cv_maf_predict()
    logging.info("Finished set up model with %d follow up iteration.", reg_data.follow_iter_)
    
    roc_result = reg_data.get_roc()
    if config_data.binary:
        r2_result = None
    else:
        r2_result = reg_data.get_r2_stats_dataframe(0.95)
    pred_dataframe = reg_data.get_per_sample_logit_mafs()
    return r2_result, roc_result, pred_dataframe, reg_data.output_metrics


if __name__ == "__main__":
    main()
