from framework.base import SKLearnModelGenerator

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class LinearDiscriminantAnalysisGenerator(SKLearnModelGenerator):
    def __init__(self):
        super(LinearDiscriminantAnalysisGenerator, self).__init__()

        solver_list = ['svd', 'lsqd', 'eigen']

        self._hp_space = [['tol', [0.000001, 0.00002], 0, None],
                          ['solver', None, 3, solver_list],
                          ['n_components', None, 1, None],
                          ['shrinkage', None, 0, None]]

        self._classifier = LinearDiscriminantAnalysis()


class QuadraticDiscriminantAnalysisGenerator(SKLearnModelGenerator):

    def __init__(self):
        super(QuadraticDiscriminantAnalysisGenerator, self).__init__()

        self._hp_space = [['tol', [0.000001, 0.00002], 0, None],
                          ['reg_param', None, 0, None]]

        self._classifier = QuadraticDiscriminantAnalysis()
