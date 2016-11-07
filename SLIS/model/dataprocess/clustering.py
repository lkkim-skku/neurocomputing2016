"""
data maining 관련 객체 또는 클래스가 정의됩니다.
"""
from model.dataprocess import *
import abc


class AbstractCluster(BaseData, metaclass=abc.ABCMeta):
    """
    Basic data mining Object

    이 Class를 inheritance할 경우, 다음의 함수를 implement해야 합니다.

    - :func:`AbstractCluster.map`

    또한 다음의 함수를 overriding하되 super의 함수를 호출해야 합니다.

    - :func:`AbstractCluster.fit`

    super의 함수를 다음과 같이 호출합니다.::

    def fit(self, x):
        super().fit(x)
        ...

    마지막으로, 현재 정의된 statistic variable은 아리와 같으니, 필요하다면 inheritance반아서 추가해야 합니다.

    - mean
    - standard derivation
    - variance
    - min
    - max
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, data):
        self._data = data
        sorted(self._data)

    @abc.abstractmethod
    def predict(self):
        return NotImplemented


class Centroid(AbstractCluster):
    """
    centroid입니다.
    centroid는 data를 대표할 수 있는 극히 적은 subdata를 의미합니다.
    보통 한 패턴당 한 개인데, 복수일 수 있습니다.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, data):
        super().fit(data)
        pass

    def predict(self):
        pass
