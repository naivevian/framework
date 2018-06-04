from framework.base import SKLearnModelGenerator

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB


class GaussianNBGenerator(SKLearnModelGenerator):

    def __init__(self):
        super(GaussianNBGenerator, self).__init__()

        self._hp_space = []
        self._classifier = GaussianNB()


class BernoulliNBGenerator(SKLearnModelGenerator):

    def __init__(self):
        super(BernoulliNBGenerator, self).__init__()

        self._hp_space = [['alpha', None, 0, None],
                          ['binarize', None, 0, None]]

        self._classifier = BernoulliNB()


class MultinomialNBGenerator(SKLearnModelGenerator):

    def __init__(self):
        super(MultinomialNBGenerator, self).__init__()

        self._hp_space = [['alpha', None, 0, None]]
        self._classifier = MultinomialNB()
