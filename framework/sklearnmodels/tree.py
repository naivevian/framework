from framework.base import SKLearnModelGenerator

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier


class DecisionTreeClassifierGenerator(SKLearnModelGenerator):

    def __init__(self):
        super(DecisionTreeClassifierGenerator, self).__init__()

        self._hp_space = [['min_samples_split', None, 0, None],
                          ['min_samples_leaf', None, 0, None],
                          ['min_weight_fraction_leaf', None, 1, None],
                          ['max_features', None, 0, None],
                          ['max_leaf_nodes', None, 0, None],
                          ['min_impurity_decrease', None, 1, None],
                          ['max_depth', None, 0, None]]

        self._classifier = DecisionTreeClassifier()


class ExtraTreeClassifierGenerator(SKLearnModelGenerator):
    def __init__(self):
        super(ExtraTreeClassifierGenerator, self).__init__()

        self._hp_space = [['min_samples_split', None, 0, None],
                          ['min_samples_leaf', None, 0, None],
                          ['min_weight_fraction_leaf', None, 1, None],
                          ['max_features', None, 0, None],
                          ['max_leaf_nodes', None, 0, None],
                          ['min_impurity_decrease', None, 1, None],
                          ['max_depth', None, 0, None]]

        self._classifier = ExtraTreeClassifier()
