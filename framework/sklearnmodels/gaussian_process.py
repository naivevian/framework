from framework.base import SKLearnModelGenerator

from sklearn.gaussian_process import GaussianProcessClassifier


class GaussianProcessClassifierGenerator(SKLearnModelGenerator):
    def __init__(self):
        super(GaussianProcessClassifierGenerator, self).__init__()

        self._hp_space = ['max_iter_predict', None, 0, None]
        # todo consider other parameters

        self._classifier = GaussianProcessClassifier()
