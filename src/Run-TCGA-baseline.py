#/usr/bin/env python3

"""
TCGA baseline model for given cancer type
"""

from pandas import read_csv, merge, DataFrame
from feather import read_dataframe
from sklearn import metrics, linear_model
from numpy import log, exp
import sys


def main():
    #logging.basicConfig()
    #logging.getLogger().setLevel(logging.INFO)
    test_type = sys.argv[1]
    if test_type == "crc":
        maf_key_ = "max_maf_pct"
    else:
        maf_key_ = "G360_max_maf_pct"

    tcga_map = {"lung": ["LUAD", "LUSC"], "breast": ["BRCA"], "bladder": ["BLCA"], "prostate": ["PRAD"], 
            "ovarian": ["OV"], "pancreatic": ["PAAD"], "gastric": ["STAD"], }

    head_path = "/ghdevhome/home/schen/epigen/ccbi-308/mcm_probe.head2.tsv"
    data_path = "/ghdevhome/home/schen/epigen/ccbi-308/mcm_probe.full_tcga.dedup.tsv"

    print("loading tcga data...")
    d_head = read_csv(head_path, sep='\t', header=0)
    colnames = d_head.columns.to_list()
    samples = colnames[1:]
    colnames.append("probe_450k")
    d_head.index = d_head["region_id"]
    d_head.pop("region_id")

    dt_msre = read_csv(data_path, sep='\t', header=None)
    dt_msre.columns = colnames
    dt_msre.index = dt_msre["region_id"].to_list()
    dt_msre.pop("region_id")

    outdir = "tcga_result/"
    roc_path = outdir + test_type + ".roc.tsv"
    pred_path = outdir + test_type + ".pred.tsv"

    print("loading TCGA...")
    meta_tumor, meta_normal = load_tcga_data(tcga_map[test_type])

    print("start modeling...")
    feature_path = "/ghdevhome/home/schen/epigen/ccbi-327/data/methylome_V2_samples.090921.tsv"
    features = read_csv(feature_path, sep='\t', header=0)
    features.loc[features.cohort=="G360_CRC", "stage_info"] = "stage_iv"

    prv_preds = get_tcga_predictions(test_type, meta_tumor, dt_msre, features, prev_as_weight, maf_key_)
    prv_fpr, prv_tpr, prv_threds = get_roc_data(prv_preds, test_type)
    prv_preds, prv_true, prv_r2 = get_maf_result(prv_preds, prv_fpr, prv_threds, test_type, prv_tpr, maf_key_)

    roc_df = DataFrame(data={"fpr": prv_fpr, "tpr": prv_tpr, "thresholds": prv_threds})
    roc_df.to_csv(roc_path, sep='\t', index=False)
    pred_df = DataFrame(data={"samples": samples, "pred": prv_preds, "true": prv_true})
    pred_df.to_csv(pred_path, sep='\t', index=False)


def get_tcga_predictions(test_type, d_meta, dt_msre, features, weight_function, maf_key_):
    df_weight = weight_function(d_meta)
    first_col_nm = df_weight.columns[0]
    dl = df_weight[first_col_nm].to_list()
    print("Total sites = %d, >=0.5 weight = %d." % (len(dl), len([x for x in dl if x >= 0.5])))
    df_weight.columns = ["weight"]
    dt_mols = dt_msre.merge(df_weight, left_on="probe_450k", right_index=True)
    print("Merged probes = %d, >=0.5 probes = %d." % (dt_mols.shape[0], dt_mols[dt_mols["weight"]>=0.5].shape[0]))

    sample_cols = dt_mols.columns.to_list()
    sample_cols.remove("weight")
    sample_cols.remove("probe_450k")
    wt_m = dt_mols[sample_cols].multiply(dt_mols["weight"], axis="index")

    tcga_preds = wt_m.sum(axis=0).to_frame()
    fcols = ["sample_id", maf_key_, "cancer_type", "somatic_call"]
    tcga_preds = tcga_preds.merge(features[fcols], left_index=True, right_on="sample_id")
    #print(tcga_preds[maf_key_])
    tcga_preds.index = tcga_preds["sample_id"].to_list()
    tcga_preds.pop("sample_id")

    tcga_preds.columns = ["pred", maf_key_, "cancer_type", "somatic_call"]
    fna = tcga_preds[(tcga_preds.somatic_call==0) & (tcga_preds.cancer_type=="cancer_free")][maf_key_].index
    tcga_preds.loc[fna, maf_key_] = tcga_preds.loc[fna, maf_key_].fillna(0)
    tcga_preds = tcga_preds[(tcga_preds.cancer_type==test_type) | (tcga_preds.cancer_type=="cancer_free")]
    
    return tcga_preds


def get_roc_data(tcga_preds, test_type):
    fpr, tpr, thresholds = metrics.roc_curve(tcga_preds["cancer_type"], tcga_preds["pred"], pos_label=test_type)
    return fpr, tpr, thresholds


def get_maf_result(tcga_preds, fpr, thresholds, test_type, tpr, maf_key_):
    min_max_maf = 1e-04
    index_95 = len(fpr[fpr<0.05])
    thred_95 = thresholds[index_95]
    print("95 spec sensi at: " + str(tpr[index_95]))

    d_mafs = tcga_preds[~tcga_preds[maf_key_].isnull()]
    if d_mafs[d_mafs["cancer_type"]==test_type].shape[0] == 0:
        print("No available MAF data for %s. Return." % test_type)
        return None, None, None
    pred_crcs = d_mafs[d_mafs["cancer_type"]==test_type]["pred"].values.reshape(-1,1)
    true_crcs = log(d_mafs[d_mafs["cancer_type"]==test_type][maf_key_] + min_max_maf).to_list()
    t_reg = linear_model.LinearRegression().fit(pred_crcs, true_crcs)
    t_reg.score(pred_crcs, true_crcs)

    m_classi_crc = tcga_preds[(~tcga_preds[maf_key_].isnull()) & (tcga_preds["pred"] >= thred_95)]
    m_pred = t_reg.predict(m_classi_crc["pred"].values.reshape(-1,1))
    m_true = log(m_classi_crc[maf_key_] + min_max_maf)
    pred_r2 = metrics.r2_score(m_true, m_pred)
    return m_pred, m_true, pred_r2


def prev_as_weight(d_meta):
    tcga_cutoff = 0.5
    cols = d_meta.columns.to_list()
    for k in ["sample_id", "sample_type", "percent_tumor_nuclei"]:
        cols.remove(k)
    dw = d_meta[cols].gt(tcga_cutoff).sum(axis=0).div(d_meta[cols].shape[0]).to_frame()
    return dw


def plain_weight(d_meta):
    cols = d_meta.columns.to_list()
    for k in ["sample_id", "sample_type", "percent_tumor_nuclei"]:
        cols.remove(k)
    dw = DataFrame(data={"X0": [1] * len(cols)}, index=cols)
    return dw


def load_tcga_data(project_names):
    meta_data = None
    for pname in project_names:
        meta_path = "/ghds/groups/lunar/data/TCGA/update/" + pname.upper()
        meta_path += "/" + pname.lower() + "_450k_meth_beta_metadata.feather"
        if meta_data is None:
            meta_data = read_dataframe(meta_path)
        else:
            d_tmp = read_dataframe(meta_path)
            meta_data = meta_data.append(d_tmp)
            
    meta_tumor = meta_data[(meta_data["sample_type"]=="Primary Tumor") & (meta_data["percent_tumor_nuclei"]>=0.5)]
    meta_normal = meta_data[meta_data["sample_type"]=="Solid Tissue Normal"]
        
    return meta_tumor, meta_normal


if __name__ == "__main__":
    main()
