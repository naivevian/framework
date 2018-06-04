"""
Author: Yi-Qi Hu
this is a template for model evaluation
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class ModelEvaluator:

    def __init__(self, model_generator=None, train_x=None, train_y=None, criterion='accuracy', valid_k=5):
        """
        :param model_generator: an instantiation son class of ModelGenerator
        :param train_x: train feature, type -- array
        :param train_y: train label, type -- array
        :param criterion: evaluation metric, type -- string
        :param valid_k: k-validation, type -- int
        """
        self.model_generator = model_generator
        self.train_x = train_x
        self.train_y = train_y
        self.criterion = criterion
        self.validation_kf = StratifiedKFold(n_splits=valid_k, shuffle=False)
        return

    @staticmethod
    def data_collector(index_list, features, labels):
        """
        re-collect data according to index
        :param index_list: the data index
        :param features: original features
        :param labels: original labels
        :return: the features and labels collected by index_list
        """

        feature_dim = features.shape[1]

        trans_features = np.zeros((len(index_list), feature_dim))
        trans_labels = np.zeros(len(index_list))

        for i in range(len(index_list)):
            trans_features[i, :] = features[index_list[i], :]
            trans_labels[i] = labels[index_list[i], :]

        return trans_features, trans_labels

    def evaluate(self, x):
        """
        evaluate the hyperparameter x by k-fold validation
        :param x: the hyperparameter list, type -- list
        :return: the evaluation value according to the metric, type -- float
        """

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
    """
    This is the father class of each model implementation. Each specific model implementation should overwrite the two
    basic functions: set_hyperparameter and generate_model.
    """

    def __init__(self):
        self._hp_space = None

    def set_hyperparameter(self, params):
        specific_params = {}

        for i in range(len(params)):
            if self._hp_space[i][2] == 0:
                specific_params[self._hp_space[i][0]] = (float(params[i]))
            elif self._hp_space[i][2] == 1:
                specific_params[self._hp_space[i][0]] = (int(params[i]))
            elif self._hp_space[i][2] == 2:
                specific_params[self._hp_space[i][0]] = params[i]  # dict, tuple params
            else:
                specific_params[self._hp_space[i][0]] = (self._hp_space[i][3][params[i]])

        return specific_params

    def generate_model(self, params):
        pass


class SKLearnModelGenerator(ModelGenerator):
    def __init__(self):
        super(SKLearnModelGenerator, self).__init__()
        self._classifier = None

    def generate_model(self, params):
        if self._classifier is None:
            raise Exception('classifier not assigned')

        classifier = self._classifier
        params_dict = self.set_hyperparameter(params)

        for item in params_dict.items():
            if hasattr(classifier, item[0]):
                setattr(classifier, item[0], item[1])

        return classifier
