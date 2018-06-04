from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC

from framework.base import SKLearnModelGenerator

kernel_list = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']


class SVCClassifier(SKLearnModelGenerator):
    def __init__(self):
        super(SVCClassifier, self).__init__()

        self._hp_space = [['C', None, 0, None],
                          ['tol', [0.000001, 0.00002], 0, None],
                          ['kernel', None, 3, kernel_list],
                          ['gamma', None, 0, None],
                          ['coef0', None, 0, None],
                          ['degree', None, 0, None]]

        self._classifier = SVC()

    # def generate_model(self, params):
    #     x = self.set_hyperparameter(params)
    #     hp_c = x[0]
    #     hp_tol = x[1]
    #     hp_kernel = x[2]
    #
    #     hp_gamma = 'auto'
    #     hp_coef0 = 0.0
    #     hp_degree = 3  # default value in sklearn
    #
    #     if hp_kernel == 'poly':
    #         hp_gamma = x[3]
    #         hp_coef0 = x[4]
    #         hp_degree = x[5]
    #     elif hp_kernel == 'rbf':
    #         hp_gamma = x[3]
    #     elif hp_kernel == 'sigmoid':
    #         hp_gamma = x[3]
    #         hp_coef0 = x[4]
    #
    #     return SVC(C=hp_c, kernel=hp_kernel, degree=hp_degree, gamma=hp_gamma, coef0=hp_coef0, tol=hp_tol)


class NuSVCClassifier(SKLearnModelGenerator):

    def __init__(self):
        super(NuSVCClassifier, self).__init__()

        self._hp_space = [['nu', [0., 1.], 0, None],
                          ['tol', [0.000001, 0.00002], 0, None],
                          ['kernel', None, 3, kernel_list],
                          ['gamma', None, 0, None],
                          ['coef0', None, 0, None],
                          ['degree', None, 0, None]]

        self._classifier = NuSVC()


class LinearSVCClassifier(SKLearnModelGenerator):

    def __init__(self):
        super(LinearSVCClassifier, self).__init__()

        loss_list = ['hinge', 'squared_hinge']
        penalty_list = ['l1', 'l2']
        self._hp_space = [['C', None, 0, None],
                          ['tol', [0.000001, 0.00002], 0, None],
                          ['loss', None, 3, loss_list],
                          ['penalty', None, 3, penalty_list]]

        self._classifier = LinearSVC()

    # def generate_model(self, params):
    #     x = self.set_hyperparameter(params)
    #     hp_c = x[0]
    #     hp_tol = x[1]
    #     hp_loss = x[2]
    #     hp_penalty = x[3]
    #
    #     return LinearSVC(penalty=hp_penalty, loss=hp_loss, tol=hp_tol, C=hp_c)
