from framework.base import SKLearnModelGenerator

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


# todo consider parameter: base_estimator
class AdaBoostClassifierGenerator(SKLearnModelGenerator):
    def __init__(self):
        super(AdaBoostClassifierGenerator, self).__init__()

        self._hp_space = [['n_estimators', None, 0, None],
                          ['learning_rate', None, 1, None]]

        self._classifier = AdaBoostClassifier()


class BaggingClassifierGenerator(SKLearnModelGenerator):
    def __init__(self):
        super(BaggingClassifierGenerator, self).__init__()

        # In sklearn, max_features' type can be float, but be int here
        self._hp_space = [['n_estimators', None, 0, None],
                          ['max_samples', None, 0, None],
                          ['max_features', None, 0, None]]

        self._classifier = BaggingClassifier()


class ExtraTreeClassifierGenerator(SKLearnModelGenerator):

    def __init__(self):
        super(ExtraTreeClassifierGenerator, self).__init__()

        # In sklearn, max_features' type can be int, float, string or None, but only int here
        self._hp_space = [['n_estimators', None, 0, None],
                          ['min_samples_split', None, 0, None],
                          ['min_samples_leaf', None, 0, None],
                          ['min_weight_fraction_leaf', None, 1, None],
                          ['max_features', None, 0, None],
                          ['max_leaf_nodes', None, 0, None],
                          ['min_impurity_decrease', None, 1, None],
                          ['max_depth', None, 0, None]]

        self._classifier = ExtraTreesClassifier()


class GradientBoostingClassifierGenerator(SKLearnModelGenerator):
    def __init__(self):
        super(GradientBoostingClassifierGenerator, self).__init__()

        # In sklearn, max_features' type can be int, float, string or None, but only int here
        self._hp_space = [['loss', None, 3, ['deviance', 'exponential']],
                          ['criterion', None, 3, ['friedman_mse', 'mse', 'mae']],
                          ['learning_rate', None, 1, None],
                          ['n_estimators', None, 0, None],
                          ['min_samples_split', None, 0, None],
                          ['min_samples_leaf', None, 0, None],
                          ['min_weight_fraction_leaf', None, 1, None],
                          ['max_features', None, 0, None],
                          ['max_leaf_nodes', None, 0, None],
                          ['min_impurity_decrease', None, 1, None],
                          ['max_depth', None, 0, None]]

        self._classifier = GradientBoostingClassifier()


class RandomForestClassifierGenerator(SKLearnModelGenerator):

    def __init__(self):
        super(RandomForestClassifierGenerator, self).__init__()

        self._hp_space = [['criterion', None, 3, ['gini', 'entropy']],
                          ['n_estimators', None, 0, None],
                          ['min_samples_split', None, 0, None],
                          ['min_samples_leaf', None, 0, None],
                          ['min_weight_fraction_leaf', None, 1, None],
                          ['max_features', None, 0, None],
                          ['max_leaf_nodes', None, 0, None],
                          ['min_impurity_decrease', None, 1, None],
                          ['max_depth', None, 0, None]]

        self._classifier = RandomForestClassifier()


class VotingClassifierGenerator(SKLearnModelGenerator):
    def __init__(self):
        super(VotingClassifierGenerator, self).__init__()

        # Estimators should use sklearn's estimators
        # Param: weights should be array-like
        self._hp_space = [['estimators', None, 2, None],
                          ['voting', None, 3, ['hard', 'soft']],
                          ['weights', None, 2, None],]

        self._classifier = VotingClassifier()