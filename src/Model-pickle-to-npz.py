import pickle, sys
from pandas import read_csv
from numpy import savez_compressed, dot

sys.path.append('/ghdevhome/home/schen/libs/SiriusEpiClassifier/src')


def get_cutoff(roc_path, fpr_level):
    rdata = read_csv(roc_path, sep='\t', header=0)
    rdata["abs_diff"] = abs(rdata["fpr"].astype('float') - fpr_level)
    rdata = rdata.sort_values(["abs_diff"])
    thred = rdata.iloc[0]["tpr"]
    if abs(thred - fpr_level) > 1e-2:
        print(roc_path, fpr_level, thred)
    return thred


def merge_pickles(z_dict, model_name, region_ids, model_prefix, roc_path):
    scaler_path = pickle_prefix + ".scaler.pkl"
    reducer_path = pickle_prefix+ ".transformer.pkl"
    predictor_path = pickle_prefix + ".predictor.pkl"
    
    infile = open(scaler_path, 'rb')
    sc = pickle.load(infile)
    infile.close()
    
    infile = open(reducer_path, 'rb')
    rd = pickle.load(infile)
    infile.close()
    
    infile = open(predictor_path, 'rb')
    preds = pickle.load(infile)
    infile.close()

    comb_mean = sc.center_ + sc.scale_ * rd.mean_
    comb_scale = sc.scale_
    new_coefs = dot(preds.mmodel.coef_, rd.components_)

    z_dict[model_name + "_region_id"] = region_ids
    z_dict[model_name + "_center_offset"] = comb_mean
    z_dict[model_name + "_weight"] = new_coefs
    z_dict[model_name + "_bias"] = preds.mmodel.intercept_
    z_dict[model_name + "threshold"] = get_cutoff(roc_path)
    z_dict[model_name + "_pseudocount"] = 0.1


def main():
    model_list_path = sys.argv[1]
    out_prefix = sys.argv[2]

    model_data = read_csv(model_list_path, sep='\t', header=0)
    mlist = model_data["model_name"].to_list()
    z_dict = {"model_list": mlist}

    count_data = read_csv(model_data.iloc[0]["count_file"])
    region_ids = count_data["region_id"].to_list()
    region_ids.remove("ctrl_sum")

    for idx, dr in model_data.iterrows():
        model_name = dr["model_name"]
        model_prefix = dr["prefix"]
        roc_path = dr["roc_path"]
        set_model_keys(z_dict, model_name, region_ids, model_prefix, roc_path)

    numpy.savez_compressed(outf, **z_dict)


if __name__ == "__main__":
    main()