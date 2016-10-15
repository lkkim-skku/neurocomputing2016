"""
data processing에 필요한 기본적인 알고리즘과 객체를 정의합니다.
:copyright: Leesuk Kim. lktime@skku.edu
"""
import numpy as np
import controller


opt_ddof = 'ddof'


def histogram(x: list, bins: int):
    """
    히스토그램
    :param x: array-like
    :param bins: bin 갯수
    :return: histogram: array-like.
    :return: steps: array-like. 각 bin의 경계값. 만약 어떤 bin이 [0, 0.5, 1]이라면 step은 [0.5, 1]임.
    """

    x = [a for a in x]
    x.sort()
    xn, xx = x[0], x[-1]
    step = (xx - xn) / bins
    box, i = xn + step, 0
    histo = []
    for a in x:
        while a > box:
            histo.append(i)
            i = 0
            box += step
        i += 1
    if len(histo) == bins:
        histo[-1] += i
    else:
        # histo.append(i)
        while len(histo) < bins:
            histo.append(i)
            i = 0
    steps = []
    box = xn + step
    for _ in range(bins):
        steps.append(box)
        box += step
    return histo, steps


def histo_cudif(x: list, bins: int):
    """
    histogram Cumulative Distribution Function

    :param x:
    :param bins:
    :return: array-like.
    """
    hist, steps = histogram(x, bins)

    k = 0.
    cdf_ = []
    samplan = len(x)
    for h in hist:
        k += h / samplan
        cdf_.append(k)

    return cdf_


class BaseDataProcess:
    """
    data process에 필요한 기본적인 정보를 정의합니다.
    """
    def __init__(self, **kwargs):
        self._mean = kwargs['mean'] if 'mean' in kwargs else 0
        self._std, self._var = kwargs['std'] if 'std' in kwargs else 0, kwargs['var'] if 'var' in kwargs else 0
        self._min, self._max = kwargs['min'] if 'min' in kwargs else 0, kwargs['max'] if 'max' in kwargs else 1
        self._opts = {opt_ddof: kwargs[opt_ddof] if opt_ddof in kwargs else 1}

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, newmean):
        self._mean = newmean

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, newstd):
        self._std = newstd

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, newvar):
        self._var = newvar

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, newmin):
        self._min = newmin

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, newmax):
        self._max = newmax


class BaseData(BaseDataProcess):
    """
    기본적으로 사용하는 데이터 모델입니다.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data = []

    @property
    def data(self):
        return self._data

    def fit(self, data: list):
        self._data = data
        self._mean = controller.call_recursive(np.mean, data)
        self._std = controller.call_recursive(np.std, data)
        self._var = controller.call_recursive(np.var, data)
        self._min = controller.call_recursive(np.min, data)
        self._max = controller.call_recursive(np.max, data)
