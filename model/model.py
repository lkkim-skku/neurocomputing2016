"""
실험을 위해 임시로 만든 model들을 정의합니다.
"""

from model.dataprocess import normalization as normaller
from model.dataprocess import clustering


class FakeCentroid(clustering.AbstractCluster):
    """
    data를 그대로 list로 감싸서 predict로 만들어버립니다.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__scalizer__ = normaller.FeatureScaler
        self.__normalizer__ = normaller.Normalizer

    def fit(self, data):
        super().fit(data)

    def predict(self):
        """
        centroid의 기본은

        1. 나눈다(clustering)

        2. feature scaling

        입니다.

        :return:
        """
        clustee = clustering.BaseData()
        clustee.fit(self._data)
        return [clustee]
        pass
