from framework.base import SKLearnModelGenerator

from sklearn.neural_network import MLPClassifier


class MLPClassifierGenerator(SKLearnModelGenerator):
    def __init__(self):
        super(MLPClassifierGenerator, self).__init__()

        activation_list = ['identity', 'logistic', 'tanh', 'relu']
        solvers = ['lbfgs', 'sgd', 'adam']
        learning_rate_options = ['constant', 'invscaling', 'adaptive']
        early_stopping = [True, False]

        self._hp_space = [['tol', [0.000001, 0.00002], 0, None],
                          ['hidden_layer_sizes', None, 2, None],
                          ['alpha', None, 0, None],
                          ['batch_size', None, 1, None],
                          ['solver', None, 3, solvers],
                          ['activation', None, 3, activation_list],
                          ['learning_rate', None, 3, learning_rate_options],
                          ['learning_rate_init', None, 0, None],
                          ['power_t', None, 0, None],
                          ['momentum', None, 0, None],
                          ['early_stopping', None, 3, early_stopping],
                          ['validation_fraction', None, 0, None],
                          ['beta_1', None, 0, None],
                          ['beta_2', None, 0, None],
                          ['epsilon', None, 0, None]]

        self._classifier = MLPClassifier()
