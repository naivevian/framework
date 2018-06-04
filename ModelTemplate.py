'''
Author: Yi-Qi Hu
this is a template for model evaluation
'''
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


class ModelEvaluater():

    def __init__(self, model_generator=None, train_x=None, train_y=None, criterion='accuracy', valid_k=5):
        '''
        :param model_generator: an instantiation son class of ModelGenerator
        :param train_x: train feature, type -- array
        :param train_y: train label, type -- array
        :param criterion: evaluation metric, type -- string
        :param valid_k: k-validation, type -- int
        '''
        self.model_generator = model_generator
        self.train_x = train_x
        self.train_y = train_y
        self.criterion = criterion
        self.validation_kf = StratifiedKFold(n_splits=valid_k, shuffle=False)
        return

    def data_collector(self, index_list, features, labels):
        '''
        re-collect data according to index
        :param index_list: the data index
        :param features: original features
        :param labels: original labels
        :return: the features and labels collected by index_list
        '''

        feature_dim = features.shape[1]

        trans_features = np.zeros((len(index_list), feature_dim))
        trans_labels = np.zeros(len(index_list))

        for i in range(len(index_list)):
            trans_features[i, :] = features[index_list[i], :]
            trans_labels[i] = labels[index_list[i], :]

        return trans_features, trans_labels

    def evaluate(self, x):
        '''
        evaluate the hyperparameter x by k-fold validation
        :param x: the hyperparameter list, type -- list
        :return: the evaluation value according to the metric, type -- float
        '''

        this_model = self.model_generator.generate_model(x)

        eval_values = []
        for train_index, valid_index in self.validation_kf.split(self.train_x, self.train_y):
            X, Y = self.data_collector(train_index, self.train_x, self.train_y)
            valid_x, valid_y = self.data_collector(valid_index, self.train_x, self.train_y)

            this_model = this_model.fit(X, Y)

            predictions = this_model.predict(valid_x)

            if self.criterion == 'accuracy':
                eval_value = accuracy_score(valid_y, predictions)
            elif self.criterion == 'auc':
                eval_value = roc_auc_score(valid_y, predictions)
            else:
                eval_value = 0.0
            eval_values.append(eval_value)

        eval_mean = np.mean(np.array(eval_values))

        return eval_mean


class ModelGenerator(object):
    '''
    This is the father class of each model implementation. Each specific model implementation should overwrite the two
    basic functions: set_hyperparameter and genrate_model.
    '''

    def __init__(self):
        return

    def set_hyperparameter(self, params):
        return

    def generate_model(self, params):
        return


# this is an example of a specific model where using sklearn package
class LogisticRegressionClassifier(ModelGenerator):

    def __init__(self):
        super(LogisticRegressionClassifier, self).__init__()

        # this is hyperparameter space for each model
        self.hp_space = [['tol', [0.000001, 0.00002], 0, None],
                         ['C', [0.01, 2], 0, None]]

    def set_hyperparameter(self, params):

        specific_params = []
        for i in range(len(params)):
            if self.hp_space[i] == 0:
                specific_params.append(float(params[i]))
            elif self.hp_space[i] == 1:
                specific_params.append(int(params[i]))
            else:
                specific_params.append(self.hp_space[i][3][params[i]])

        return specific_params

    def generate_model(self, params):

        x = self.set_hyperparameter(params)

        hp_tol = x[0]
        hp_c = x[1]

        this_classifier = LogisticRegression(tol=hp_tol, C=hp_c)

        return this_classifier


# this is an example of a specific model where using self-defined training model
class SelfDefinedClassifier(ModelGenerator):

    def __init__(self):
        super(SelfDefinedClassifier, self).__init__()
        return

    def set_hyperparameter(self, params):
        return params

    def generate_model(self, params):
        this_classifier = SelfDefined(param1=params[0], param2=params[1])
        return this_classifier


# the implementation of self-defined model, training and prediction
class SelfDefined():

    def __init__(self, param1=0, param2=0):
        return

    def fit(self, X, Y):
        '''
        implement the training process here
        :param X: train feature, type -- array
        :param Y: train label, type -- array
        :return: the instantiation of this class
        '''
        return self

    def predict(self, X):
        '''
        implement the prediction process here
        :param X: test feature, type -- array
        :return: prediction, type -- array
        '''
        return np.zeros((X.shape[0]))
