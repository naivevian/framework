from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron

from framework.base import SKLearnModelGenerator


class LogisticRegressionClassifierGenerator(SKLearnModelGenerator):

    def __init__(self):
        super(LogisticRegressionClassifierGenerator, self).__init__()

        # this is hyperparameter space for each model
        self._hp_space = [['tol', [0.000001, 0.00002], 0, None],
                          ['C', [0.01, 2], 0, None]]

        self._classifier = LogisticRegression()


class SGDClassifierGenerator(SKLearnModelGenerator):

    def __init__(self):
        super(SGDClassifierGenerator, self).__init__()
        loss_functions = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron',
                          'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']

        penalty = ['none', 'l2', 'l1', 'elasticnet']

        learning_rate_options = ['constant', 'optimal', 'invascaling']

        self._hp_space = [['tol', [0.000001, 0.00002], 0, None],
                          ['alpha', None, 0, None],
                          ['loss', None, 3, loss_functions],
                          ['penalty', None, 3, penalty],
                          ['l1_ratio', [0., 1.], 0, None],
                          ['epsilon', None, 0, None],
                          ['learning_rate', None, 3, learning_rate_options]]

        self._classifier = SGDClassifier()


class RidgeClassifierGenerator(SKLearnModelGenerator):
    def __init__(self):
        super(RidgeClassifierGenerator, self).__init__()

        solvers = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

        self._hp_space = [['tol', [0.000001, 0.00002], 0, None],
                          ['alpha', None, 0, None],
                          ['solver', None, 3, solvers]]

        self._classifier = RidgeClassifier()


class PassiveAggressiveClassifierGenerator(SKLearnModelGenerator):

    def __init__(self):
        super(PassiveAggressiveClassifierGenerator, self).__init__()

        loss_functions = ['hinge', 'squared_hinge']

        self._hp_space = [['tol', [0.000001, 0.00002], 0, None],
                          ['C', None, 0, None],
                          ['loss', None, 3, loss_functions]]

        self._classifier = PassiveAggressiveClassifier()


class PerceptronGenerator(SKLearnModelGenerator):
    def __init__(self):
        super(PerceptronGenerator, self).__init__()

        penalty_list = ['l1', 'l2']

        self._hp_space = [['tol', [0.000001, 0.00002], 0, None],
                          ['alpha', None, 0, None],
                          ['eta0', None, 0, None],
                          ['penalty', None, 3, penalty_list]]

        self._classifier = Perceptron()
