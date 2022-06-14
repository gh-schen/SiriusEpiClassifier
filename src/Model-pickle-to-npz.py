import pickle, sys
from pandas import read_csv
from numpy import savez_compressed, dot

sys.path.append('/ghdevhome/home/schen/libs/SiriusEpiClassifier/src')


def get_cutoff(roc_path, spec_level):
    rdata = read_csv(roc_path, sep='\t', header=0)
    rdata["abs_diff"] = abs(rdata["specificity"].astype('float') - spec_level)
    rdata = rdata.sort_values(["abs_diff"])
    thred = rdata.iloc[0]["cutoff"]
    if abs(rdata.iloc[0]["specificity"] - spec_level) > 1e-2:
        print(roc_path, spec_level, thred)
    return thred


def set_model_keys(z_dict, model_name, region_ids, model_prefix, roc_path):
    scaler_path = model_prefix + ".scaler.pkl"
    reducer_path = model_prefix+ ".transformer.pkl"
    predictor_path = model_prefix + ".predictor.pkl"
    
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
    z_dict[model_name + "_scale_offset"] = comb_scale
    z_dict[model_name + "_center_offset"] = comb_mean
    z_dict[model_name + "_weight"] = new_coefs.ravel()
    if model_name.endswith("lr"): # update LR threshold to 0
        real_bias = preds.mmodel.intercept_[0]
        threshold = get_cutoff(roc_path, 0.98)
        adjust_bias = real_bias - threshold
        z_dict[model_name + "_bias"] = adjust_bias
        #print(real_bias, adjust_bias)
        z_dict[model_name + "_threshold"] = 0
    else:
        z_dict[model_name + "_bias"] = preds.mmodel.intercept_
        z_dict[model_name + "_threshold"] = get_cutoff(roc_path, 0.98)
    z_dict[model_name + "_pseudocount"] = 1e-06

    #for i in ["_bias", "_threshold", "_pseudocount"]:
    #    print(z_dict[model_name + i])


def main():
    model_list_path = sys.argv[1]
    out_prefix = sys.argv[2]

    model_data = read_csv(model_list_path, sep='\t', header=0)
    mlist = model_data["model_name"].to_list()
    z_dict = {"model_list": mlist}

    count_data = read_csv(model_data.iloc[0]["region_list_file"])
    region_ids = count_data["region_id"].to_list()

    for idx, dr in model_data.iterrows():
        model_name = dr["model_name"]
        model_prefix = dr["prefix"]
        roc_path = dr["roc_path"]
        #print(model_name, model_prefix, roc_path)
        set_model_keys(z_dict, model_name, region_ids, model_prefix, roc_path)

    savez_compressed(out_prefix, **z_dict)


if __name__ == "__main__":
    main()
