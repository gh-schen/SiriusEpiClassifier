from numpy.core.fromnumeric import transpose
from sklearn import linear_model
from scipy.special import logit
from scipy import stats
from copy import deepcopy
from numpy import random, concatenate, quantile, matmul, transpose
import logging


class singleRegModel():
    """
    data struct for running a single regression test
    """
    def __init__(self, regressor):
        self.regressor = regressor
        self.mmodel = None
        # params
        self.quantile_limit_ = 0.95


    def train_binary(self, x_train, y_train):
        self.mmodel = deepcopy(self.regressor)
        self.mmodel.fit(x_train, y_train)


    def train_quant(self, init_x, follow_x, init_y, follow_iter):
        self.train_binary(init_x, init_y)
        if follow_x is None:
            logging.warning("No samples have missing MAF - no follow up training")
            return

        for i in range(follow_iter):
            init_preds = self.mmodel.predict(init_x)
            upper_limit = quantile(init_preds, self.quantile_limit_)
            follow_y = self.mmodel.predict(follow_x)
            follow_y[follow_y > upper_limit] = upper_limit

            x_merge = concatenate((init_x, follow_x))
            y_merge = concatenate((init_y, follow_y))

            self.mmodel = deepcopy(self.regressor)
            self.mmodel.fit(x_merge, y_merge)


    def predict_prob(self, input_x):
        preds = matmul(input_x, transpose(self.mmodel.coef_)) + self.mmodel.intercept_
        probs = preds[:,0]
        return probs


    def predict_quant(self, input_x):
        return self.mmodel.predict(input_x)
