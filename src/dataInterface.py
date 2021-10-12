from statistics import median, mean
from pandas import read_csv, merge, DataFrame
from numpy import nan
import json


def read_features(feature_path, bad_cohorts, bad_batches):
    features = read_csv(feature_path, sep='\t', header=0)

    # merge cohort names
    raw_names = list(set(features.cohort.to_list()))
    cohort_map = {}
    for r in raw_names:
        cohort_map[r] = r.split('_')[0]
        
    features = features.replace({"cohort": cohort_map})
    features = features[~features.cohort.isin(bad_cohorts)]
    features = features[~features.batch.isin(bad_batches)]
    return features


def load_molcounts_data(fname, features, cancer_name, maf_key):
    mdata = read_csv(fname, sep='\t', header=0)
    mdata = mdata.T
    mdata.columns = mdata.iloc[0].to_list()
    mdata = mdata.iloc[1:]
    region_list = mdata.columns.to_list()
    region_list.remove("ctrl_sum")
    
    extra_keys = [maf_key, "somatic_call", "cancer_type", "cohort", "stage", "sample_id"]
    mdata = merge(mdata, features[extra_keys], left_index=True, right_on="sample_id")
    mdata.index = mdata["sample_id"].to_list()
    
    mdata[maf_key] = mdata[maf_key].div(100)
    mdata = mdata.rename(columns = {maf_key: 'maf'})
    
    extra_keys[0] = "maf"
    extra_keys = extra_keys[:-1]
    extra_keys.append("ctrl_sum")
    
    mdata["cancer_type"] = mdata["cancer_type"].str.lower()
    tumor_data =  mdata[mdata.cancer_type.isin([cancer_name, "cancer_free"])][region_list + extra_keys]
    return tumor_data, region_list


def set_roc(roc_map, reg_roc, num_digits):
    # add ROC result from one single run
    for idx, dt in reg_roc.iterrows():
        fpr = round(dt["fpr"], num_digits)
        tpr = dt["tpr"]
        if fpr not in roc_map:
            roc_map[fpr] = [[], []]
        roc_map[fpr][0].append(tpr)
        roc_map[fpr][1].append(dt["cutoff"])


def convert_roc_map_to_dataframe(roc_map, num_digits):
    fpr_sorted = sorted(list(roc_map.keys()), reverse=True)
    roc_result = DataFrame(data={"specificity": [], "min": [], "max": [], "mean": [], "median": [], "cutoff": [], "num_points": []})
    for fval in fpr_sorted:
        sensis = roc_map[fval][0]
        cutoffs = roc_map[fval][1]
        ds = [1-fval, min(sensis), max(sensis), mean(sensis), median(sensis), median(cutoffs), len(sensis)]
        ds[:-1] = [round(x, num_digits) for x in ds[:-1]]
        roc_result.loc[roc_result.shape[0]] = ds
    return roc_result


def dump_prediction_result(output_prefix, final_roc, final_r2, final_pred, final_metrics):
    final_roc.to_csv(output_prefix + ".roc.tsv", sep='\t', index=False)
    if final_r2 is not None:
        final_r2.to_csv(output_prefix + ".r2.tsv", sep='\t', index=True)
    final_pred.to_csv(output_prefix + ".pred.tsv", sep='\t', index=True)
    outfile = open(output_prefix + ".metrics.json", 'w')
    json.dump(final_metrics, outfile)
    outfile.close()


def gen_simulated_data(indata, region_list, params, cancer_type):
    # generate in silico titration data to 0.05% MAF
    # for now not using params
    target_maf = 5e-4
    sample_seed = 100
    dt_tumor = indata[(indata.cancer_type==cancer_type) & (~indata.maf.isnull())]
    num_tumors = dt_tumor.shape[0]
    dt_normal = indata[indata.cancer_type!=cancer_type].sample(n=num_tumors, random_state=sample_seed)
    coef_tumor = target_maf / dt_tumor.maf
    coef_normal = 1 - coef_tumor
    dt_simu = (dt_tumor[region_list] * coef_tumor) + (dt_normal[region_list] * coef_normal)
    dt_simu["ctrl_sum"] = (dt_tumor["ctrl_sum"] * coef_tumor) + (dt_normal["ctrl_sum"] * coef_normal)
    dt_simu["sample_id"] = dt_tumor["sample_id"] + "_simu"
    dt_simu["cancer_type"] = cancer_type
    dt_simu["maf"] = target_maf
    for k in ["somatic_call", "cohort", "stage"]:
        dt_simu[k] = dt_tumor[k]

    return dt_simu
