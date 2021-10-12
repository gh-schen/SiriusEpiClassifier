from mafUtility import singleRegModel, predOutcome
from pandas import DataFrame
from statistics import median, mean
from numpy import log, concatenate
from sklearn import linear_model, preprocessing, metrics, decomposition
from scipy.special import logit, expit
from copy import deepcopy

import logging


class regData():
    """
    data struct for running CV regression - use singleRegModel as its core
    """
    def __init__(self, params):
        # mode
        self.training_only = False
        self.test_only = False
        # params
        self.min_maf_ = 1e-06
        self.x_offset_ = 0.1
        self.num_cv_ = 4
        self.cancer_type_str_ = params.cancer_type
        self.cancer_free_str_ = "cancer_free"
        self.label_key_ = "cancer_type"
        self.ctrl_key_ = "ctrl_sum"
        self.maf_key_ = "maf"
        self.somatic_cleanup = params.somatic_cleanup
        #self.intercept_key_ = "intercept"
        self.min_total_pos_ctrl_ = 1000
        self.follow_iter_ = 1 # number of iterations for training data points with no MAF
        self.total_explained_variance_ = 0.9 # total variance explained
        self.num_components_list = [0] * self.num_cv_ # finally how many components were used
        # data
        self.init_train_x = [] # partitions of training x
        self.init_indexes = []
        self.follow_train_x = []
        self.follow_train_indexes = []
        self.init_train_y = []
        self.follow_train_labels = []
        self.test_x = []
        self.test_indexes = []
        self.test_y = []
        self.follow_test_x = []
        self.follow_test_indexes = []
        # data params
        self.do_clean_up_ = params.do_clean_up
        self.do_transform_ = params.do_transform
        self.min_norm_count_in_max_ = 2e-06
        self.tumor_normal_ratio_min_ = params.tumor_normal_ratio_min
        self.scaler_ = None
        if params.scaler_str:
            self.scaler_ = eval(params.scaler_str)
        self.scale_model = None
        self.pca_model = None
        # model
        self.is_binary_classifier_ = params.binary
        self.regressor_ = eval(params.regressor_str)
        self.trained_model = None
        # result
        self.pred_map = {}
        self.roc_dataframe = None
        # result metrics
        self.output_metrics = {"num_components": [], "num_features_after_clean_up": None}
        

    def _set_split_data(self, rawdata, regions, num_partitions, maf_exist):
        pnum = round(rawdata.shape[0] / num_partitions) + 1
        if rawdata.shape[0] == 0:
            if self.test_only:
                return
            raise Exception("Empty input data in classifier.")

        new_x = (rawdata[regions] + self.x_offset_).div(rawdata[self.ctrl_key_].values, axis=0)
        new_x = log(new_x.astype('float'))
        if self.is_binary_classifier_:
            new_y = rawdata[self.label_key_]
            new_y = new_y.replace(self.cancer_type_str_, 1)
            new_y = new_y.replace(self.cancer_free_str_, 0)
        else:
            new_y = logit(rawdata[self.maf_key_].fillna(self.min_maf_))

        pstart = 0
        pindex = 0
        while(pstart < rawdata.shape[0]):
            pstop = min(pstart + pnum, rawdata.shape[0])
            test_locs = list(range(pstart, pstop))
            all_locs = list(range(rawdata.shape[0]))
            train_locs = list(set(all_locs) ^ set(test_locs))
            if self.num_cv_ == 1:
                train_locs = test_locs
            x_train = new_x.iloc[train_locs]
            x_test = new_x.iloc[test_locs]
            y_train = new_y.iloc[train_locs]
            y_test = new_y.iloc[test_locs]
            if maf_exist: # initial train/test
                if len(self.init_train_x) <= pindex: # this partition has no data yest
                    self.init_train_x.append(x_train)
                    self.init_indexes.append(x_train.index.to_list())
                    self.init_train_y.append(y_train)
                    self.test_x.append(x_test)
                    self.test_indexes.append(x_test.index.to_list())
                    self.test_y.append(y_test)
                else: # add to existing data partition
                    self.init_train_x[pindex] = self.init_train_x[pindex].append(x_train)
                    self.init_indexes[pindex] += x_train.index.to_list()
                    self.init_train_y[pindex] = self.init_train_y[pindex].append(y_train)
                    self.test_x[pindex] = self.test_x[pindex].append(x_test)
                    self.test_indexes[pindex] += x_test.index.to_list()
                    self.test_y[pindex] = self.test_y[pindex].append(y_test)
            else: # no maf info -> add to follows
                self.follow_train_x.append(x_train)
                self.follow_test_x.append(x_test)
                self.follow_train_indexes.append(x_train.index.to_list())
                self.follow_test_indexes.append(x_test.index.to_list())
                self.follow_train_labels.append(y_train)
            pindex += 1
            pstart = pstop


    def _transform_features(self, raw_init, raw_follow, raw_test, raw_follow_test):
        """
        For now do PCA on normalized counts
        """
        if not self.test_only:
            if raw_follow is not None:
                d_pca = concatenate((raw_init, raw_follow))
            else:
                d_pca = raw_init
            init_pca = decomposition.PCA()
            init_pca.fit(d_pca)
            total_var = 0
            num_comp = 0
            for v in init_pca.explained_variance_ratio_:
                total_var += v
                num_comp += 1
                if total_var >= self.total_explained_variance_:
                    break
            self.output_metrics["num_components"].append(num_comp)
            self.pca_model = decomposition.PCA(n_components=num_comp)
            self.pca_model.fit(d_pca)

        t_init = self.pca_model.transform(raw_init)
        t_test = self.pca_model.transform(raw_test)
        if raw_follow is not None:
            t_follow = self.pca_model.transform(raw_follow)
            t_follow_test = self.pca_model.transform(raw_follow_test)
        else:
            t_follow = None
            t_follow_test = None
        return t_init, t_follow, t_test, t_follow_test


    def _clean_input_data(self, count_data, raw_regions):
        min_norm_val = 1e-10
        # filter on max counts
        dt_tumor = count_data[count_data[self.label_key_] == self.cancer_type_str_]
        dt_tumor = dt_tumor[raw_regions].div(dt_tumor[self.ctrl_key_].values, axis=0)
        count_filter = dt_tumor[raw_regions].max(axis=0) >= self.min_norm_count_in_max_
        tmp_regions = dt_tumor[raw_regions].columns[count_filter]
        dt_tumor = dt_tumor[tmp_regions]

        # remove those with normal > tumor (threshold varies)
        dt_normal = count_data[count_data[self.label_key_] == self.cancer_free_str_]
        dt_normal = dt_normal[tmp_regions].div(dt_normal[self.ctrl_key_].values, axis=0)

        wt_tumor = dt_tumor.max(axis=0)
        wt_tumor.columns = ["weight"]
        wt_normal = dt_normal.median(axis=0)
        wt_normal.columns = ["weight"]
        wt_normal = wt_normal.replace(0, min_norm_val)

        dt_merge = wt_tumor.div(wt_normal)
        new_regions = dt_merge[dt_merge >= self.tumor_normal_ratio_min_].index.to_list()

        removed_cols = set(new_regions) ^ set(raw_regions)
        kept_cols = list(set(count_data.columns.to_list()) ^ removed_cols)
        new_data = count_data[kept_cols]
        return new_data, new_regions


    def _normalize_input_data(self):
        for ii in range(self.num_cv_):
            if len(self.follow_train_x) > ii:
                d_train = concatenate((self.init_train_x[ii], self.follow_train_x[ii]))
            else:
                d_train = self.init_train_x[ii]
            if not self.test_only:
                self.scale_model = deepcopy(self.scaler_)
                self.scale_model.fit(d_train)
                self.init_train_x[ii] = self.scale_model.transform(self.init_train_x[ii])
                if len(self.follow_train_x) > ii:
                    self.follow_train_x[ii] = self.scale_model.transform(self.follow_train_x[ii])
            self.test_x[ii] = self.scale_model.transform(self.test_x[ii])
            if len(self.follow_train_x) > ii:
                self.follow_test_x[ii] = self.scale_model.transform(self.follow_test_x[ii])


    def _transform_input_data(self):
        for ii in range(self.num_cv_):
            if ii < len(self.follow_train_x):
                t_init_train, t_follow_train, t_init_test, t_follow_test = self._transform_features(
                    self.init_train_x[ii], self.follow_train_x[ii], self.test_x[ii], self.follow_test_x[ii])
            else:
                t_init_train, t_follow_train, t_init_test, t_follow_test = self._transform_features(
                    self.init_train_x[ii], None, self.test_x[ii], None)
            self.init_train_x[ii] = t_init_train
            self.test_x[ii] = t_init_test
            if ii < len(self.follow_train_x):
                self.follow_train_x[ii] = t_follow_train
                self.follow_test_x[ii] = t_follow_test


    def set_cv_data(self, count_data, input_regions, shuffle_seed):
        """
        Prepare CV by partitioning & transforming data
        """
        if self.training_only or self.test_only: # no CV in training mode
            self.num_cv_ = 1
            self.do_clean_up_ = False # temporarily not doing clean up for separate train & test

        if self.ctrl_key_ not in count_data.columns:
            raise Exception("Need to have the control key %s" % self.ctrl_key_)

        indata = count_data[count_data[self.ctrl_key_] > self.min_total_pos_ctrl_].sort_index()

        if self.do_clean_up_:
            indata, regions = self._clean_input_data(indata, input_regions)
            indata = indata.sample(frac=1, random_state=shuffle_seed)
            self.output_metrics["num_features_after_clean_up"] = len(regions)
        else:
            regions = input_regions

        # then set training
        if self.somatic_cleanup:
            init_cancer_index = (indata[self.label_key_] == self.cancer_type_str_) & (indata["somatic_call"] == 1)
            init_normal_index = (indata[self.label_key_] == self.cancer_free_str_) & (indata["somatic_call"] == 0)
        else:
            init_cancer_index = (indata[self.label_key_] == self.cancer_type_str_) & (~indata[self.maf_key_].isnull())
            init_normal_index = (indata[self.label_key_] == self.cancer_free_str_)

        init_cancer = indata[init_cancer_index]
        init_normal = indata[init_normal_index]
        follows = indata[~(init_cancer_index | init_normal_index)]

        self._set_split_data(init_cancer, regions, self.num_cv_, maf_exist=True)
        self._set_split_data(init_normal, regions, self.num_cv_, maf_exist=True)
        if follows.shape[0] > 0:
            self._set_split_data(follows, regions, self.num_cv_, maf_exist=False)
        else:
            logging.warning("All cancer samples have MAF - this is unusual.")

        if self.scaler_ is not None:
            self._normalize_input_data()

        # set up samples
        for ii in range(self.num_cv_):
            tlist = self.test_indexes[ii]
            for j in range(len(tlist)):
                po = predOutcome()
                po.true_y = self.test_y[ii][j]
                self.pred_map[tlist[j]] = po
            if ii < len(self.follow_test_indexes):
                tlist = self.follow_test_indexes[ii]
                for j in range(len(tlist)):
                    po = predOutcome()
                    po.true_y = None
                    self.pred_map[tlist[j]] = po

        # transform features
        if self.do_transform_:
            self._transform_input_data()

        for s in init_cancer.index:
            self.pred_map[s].cancer_status = 1
        for s in init_normal.index:
            self.pred_map[s].cancer_status = 0
        for dname, dinfo in follows.iterrows():
            ct = dinfo[self.label_key_]
            if ct == self.cancer_type_str_:
                cstatus = 1
            elif ct == self.cancer_free_str_:
                cstatus = 0
            else:
                continue
            self.pred_map[dname].cancer_status = cstatus


    def set_simu_data(self, count_data, input_regions):
        # add simulated data to training only


    def _run_binary_training(self, srm, iter_index):
        if len(self.follow_train_x) > 0:
            x_train = concatenate((self.init_train_x[iter_index], self.follow_train_x[iter_index]))
            y_train = concatenate((self.init_train_y[iter_index], self.follow_train_labels[iter_index]))
        else:
            x_train = self.init_train_x[iter_index]
            y_train = self.init_train_y[iter_index]
        srm.train_binary(x_train, y_train)
        train_y = srm.predict_prob(x_train)
        if len(self.follow_train_x) > 0:
            tlist = self.init_indexes[iter_index] + self.follow_train_indexes[iter_index]
        else:
            tlist = self.init_indexes[iter_index]
        for j in range(len(tlist)):
            self.pred_map[tlist[j]].train_ys.append(train_y[j])


    def _run_binary_prediction(self, srm, iter_index):
        self._run_binary_training(srm, iter_index)

        if len(self.follow_test_x) > 0:
            x_test = concatenate((self.test_x[iter_index], self.follow_test_x[iter_index]))
            tlist = self.test_indexes[iter_index] + self.follow_test_indexes[iter_index]
        else:
            x_test = self.test_x[iter_index]
            tlist = self.test_indexes[iter_index]
        test_y = srm.predict_prob(x_test)
        for j in range(len(tlist)):
            self.pred_map[tlist[j]].test_y = test_y[j]


    def _run_quant_training(self, srm, iter_index):
        if len(self.follow_train_x) > 0:
            f_train = self.follow_train_x[iter_index]
        else:
            f_train = None
        srm.train_quant(self.init_train_x[iter_index], f_train, self.init_train_y[iter_index], self.follow_iter_)
        train_y = srm.predict_quant(self.init_train_x[iter_index])
        tlist = self.init_indexes[iter_index]
        for j in range(len(tlist)):
            self.pred_map[tlist[j]].train_ys.append(train_y[j])
        # follow up train
        if len(self.follow_train_x) > 0:
            train_y = srm.predict_quant(self.follow_train_x[iter_index])
            tlist = self.follow_train_indexes[iter_index]
            for j in range(len(tlist)):
                self.pred_map[tlist[j]].train_ys.append(train_y[j])


    def _run_quant_prediction(self, srm, iter_index):
        self._run_quant_training(srm, iter_index)

        # init test
        test_y = srm.predict_quant(self.test_x[iter_index])
        tlist = self.test_indexes[iter_index]
        for j in range(len(tlist)):
            self.pred_map[tlist[j]].test_y = test_y[j]

        # follow up test
        if len(self.follow_test_indexes) > 0:
            test_y = srm.predict_quant(self.follow_test_x[iter_index])
            tlist = self.follow_test_indexes[iter_index]
            for j in range(len(tlist)):
                self.pred_map[tlist[j]].test_y = test_y[j]


    def run_cv_maf_predict(self):
        for ii in range(self.num_cv_):
            srm = singleRegModel(self.regressor_)
            if self.is_binary_classifier_:
                self._run_binary_prediction(srm, ii)
            else:
                self._run_quant_prediction(srm, ii)


    def run_training(self):
        srm = singleRegModel(self.regressor_)
        self.trained_model = srm
        if self.is_binary_classifier_:
            self._run_binary_training(srm, 0)
        else:
            self._run_quant_training(srm, 0)


    def run_predict_only(self):
        if self.follow_test_x:
            x_test = concatenate((self.test_x[0], self.follow_test_x[0]))
            tlist = self.test_indexes[0] + self.follow_test_indexes[0]
        else:
            x_test = self.test_x[0]
            tlist = self.test_indexes[0]
        if self.is_binary_classifier_:
            test_y = self.trained_model.predict_prob(x_test)
        else:
            test_y = self.trained_model.predict_quant(x_test)
        for j in range(len(tlist)):
            self.pred_map[tlist[j]].test_y = test_y[j]


    def get_roc(self, rtype="test"):
        """
        return roc curve
        """
        test_ys = []
        cancer_stats = []
        for k,v in self.pred_map.items():
            if v.cancer_status is None:
                continue
            if rtype == "test":
                test_ys.append(v.test_y)
            elif rtype == "train":
                test_ys.append(median(v.train_ys))
            else:
                raise Exception("Unable to recognize roc type: %s.", rtype)
            cancer_stats.append(v.cancer_status)

        fpr, tpr, threds = metrics.roc_curve(cancer_stats, test_ys, pos_label=1)
        self.roc_dataframe = DataFrame(data={"fpr": fpr, "tpr": tpr, "cutoff": threds})
        return self.roc_dataframe


    def get_per_sample_logit_mafs(self):
        if self.roc_dataframe is None:
            raise Exception("Run get_roc first before getting per-sample logit!")

        true_ys = []
        test_ys = []
        samples = []
        train_ys = []
        states = []
        for k,v in self.pred_map.items():
            samples.append(k)
            true_ys.append(v.true_y)
            test_ys.append(v.test_y)
            if not self.test_only:
                train_ys.append(median(v.train_ys))
            states.append(v.cancer_status)
        pred_dataframe = DataFrame(data={"samples": samples, "true": true_ys, "pred": test_ys, "status": states})
        if self.test_only:
            pred_dataframe["train"] = [0] * pred_dataframe.shape[0]
        else:
            pred_dataframe["train"] = train_ys
        return pred_dataframe


    def get_r2_stats_dataframe(self, spec_cutoff):
        if self.roc_dataframe is None:
            raise Exception("Run get_roc first before getting R2!")

        closest_fpr = 1 - spec_cutoff
        df = self.roc_dataframe
        # get the roc cutoff at input spec_cutoff
        df["abs_diff"] = abs(df["fpr"].astype('float') - closest_fpr)
        df = df.sort_values(["abs_diff"])
        logit_cutoff = df.iloc[0]["cutoff"]

        residuals_logit = []
        residuals_real = []
        true_ys_real = []
        test_ys_real = []
        true_ys_logit = []
        test_ys_logit = []
        num_pos = 0
        for k,v in self.pred_map.items():
            if v.cancer_status is None:
                continue
            if v.test_y >= logit_cutoff:
                num_pos += 1
                if v.true_y is not None:
                    true_ys_logit.append(v.true_y)
                    test_ys_logit.append(v.test_y)
                    residuals_logit.append(v.test_y - v.true_y)
                    true_ys_real.append(expit(v.true_y))
                    test_ys_real.append(expit(v.test_y))
                    residuals_real.append(expit(v.test_y) - expit(v.true_y))

        r2_result = DataFrame(data={"r2": [], "mean_residual": [], "median_residual": [], "num_positive": [], "cutoff": []})
        r2_val = metrics.r2_score(true_ys_logit, test_ys_logit)
        r2_result.loc["logit"] = [r2_val, mean(residuals_logit), median(residuals_logit), num_pos, logit_cutoff]
        r2_val = metrics.r2_score(true_ys_real, test_ys_real)
        r2_result.loc["real"] = [r2_val, mean(residuals_real), median(residuals_real), num_pos, expit(logit_cutoff)]
        return r2_result


