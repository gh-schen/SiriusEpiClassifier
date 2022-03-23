import pickle, sys
from numpy import savez_compressed, dot

sys.path.append('/ghdevhome/home/schen/libs/SiriusEpiClassifier/src')

def merge_pickles(pickle_prefix, out_prefix):
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

    outf = out_prefix
    savez_compressed(outf, region_id=map(str, list(range(comb_mean.shape[0]))), center_offset=comb_mean,
                     scale_offset=comb_scale, coefs=new_coefs[0], bias=preds.mmodel.intercept_[0])


def main():
    pickle_prefix = sys.argv[1]
    out_prefix = sys.argv[2]
    merge_pickles(pickle_prefix, out_prefix)


if __name__ == "__main__":
    main()