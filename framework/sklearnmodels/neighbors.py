from framework.base import SKLearnModelGenerator

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier


class KNeighborsClassifierGenerator(SKLearnModelGenerator):

    def __init__(self):
        super(KNeighborsClassifierGenerator, self).__init__()

        self._hp_space = [['n_neighbors', None, 1, None]]
        self._classifier = KNeighborsClassifier()


class RadiusNeighborsClassifierGenerator(SKLearnModelGenerator):

    def __init__(self):
        super(RadiusNeighborsClassifierGenerator, self).__init__()

        self._hp_space = [['radius', None, 0, None]]
        self._classifier = RadiusNeighborsClassifier()
