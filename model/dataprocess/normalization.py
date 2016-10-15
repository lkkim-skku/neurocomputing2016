"""
normalization, kernel 등을 정의합니다.

Normalizer
FeatureScaler
student
logscaling
kernel
kernel_weight
kernel_gaussian
kernel_epanechnikov
distance_euclidean
distance_mahalanobis
"""
from model.dataprocess import *
import math
import abc
import controller


class AbstractMapper(BaseDataProcess, metaclass=abc.ABCMeta):
    """
    mapping하기 위해 필요한 기반재료를 정의합니다.

    이 Class를 inheritance할 경우, 다음의 함수를 implement해야 합니다.

    - :func:`AbstractMapper.map`

    또한 다음의 함수를 overriding하되 super의 함수를 호출해야 합니다.

    - :func:`AbstractMapper.fit`

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

    def fit(self, data):
        self._mean = controller.call_recursive(np.mean, data)
        self._std = controller.call_recursive(np.std, data)
        self._var = controller.call_recursive(np.var, data)
        self._min = controller.call_recursive(np.min, data)
        self._max = controller.call_recursive(np.max, data)
        # self._mean = np.mean(data)
        # self._std = np.std(data, ddof=self._opts[opt_ddof])
        # self._var = np.var(data, ddof=self._opts[opt_ddof])
        # self._min, self._max = min(data), max(data)

    @abc.abstractmethod
    def map(self, x):
        raise NotImplementedError
    pass


class Normalizer(AbstractMapper):
    """
    Eucliean Normalization을 합니다. 아마도.
    n차원의 data를 1차원으로 mapping합니다.
    """

    def fit(self, data: list):
        super().fit(data)
        n = [self.map(x) for x in data]
        return n

    def map(self, x):
        return distance_euclidean(x, self._mean)


class FeatureScaler(AbstractMapper):
    """
    feature scaling을 전담하는 객체입니다.
    """

    def fit(self, data: list or tuple):
        """
        fit하고 fit하는 대상의 data를 map해서 리턴
        :param data:
        :return:
        """
        super().fit(data)
        fs = [self.map(x) for x in data]
        sorted(fs)
        return fs

    def map(self, x):
        """
        feature-scaling합니다.

        :param x: numeric.
        :return: numeric. scaled value.
        """
        return 0 if self._max == self._min or x == self._min \
            else (x - self._min) / (self._max - self._min)

    pass


# def formula_featurescaling(x, x_min, x_max):
#     return 0 if x_max == x_min or x == x_min else (x - x_min) / (x_max - x_min)
#
#
# def featurescaling(x, **kwargs):
#     """
#     kwargs가 없고 x가 list라면 featurescaling하면서 x의 min과 max를 리턴
#     kwargs가 있으면 거기서 xmin과 xmax를 가져오면서 featurescaling
#
#     :param x:
#     :param kwargs:
#     :return:
#     """
#
#     if 'x_min' in kwargs and 'x_max' in kwargs:
#         return x if type(x) == str else formula_featurescaling(x, kwargs['x_min'], kwargs['x_max'])
#     else:
#         x_min, x_max = min(x), max(x)
#         x_fs = [formula_featurescaling(a, x_min, x_max) for a in x]
#         return x_fs, x_min, x_max


# def formula_student(x, mean, std):
#     return (x - mean) / std


class Student(AbstractMapper):
    """
    Student's stastistic
    """

    def map(self, x):
        return (x - self._mean) / self._std

    pass


# def student(x, **kwargs):
#     """
#     Student's statistic
#
#     :param x:
#     :param kwargs:
#     :return:
#     """
#     if 'mean' in kwargs and 'std' in kwargs:
#         return x if type(x) == str else formula_student(x, kwargs['mean'], kwargs['std'])
#     else:
#         m, s = np.mean(x), np.std(x, ddof=1)
#         x_ss = [formula_student(a, m, s) for a in x]
#         return x_ss, m, s


def logscaling(x):
    """
    log scaling

    :param x:
    :return:
    """
    return math.log(1 + x) / math.log(2)


opt_weight = 'weight'
opt_mean = 'mean'
opt_std = 'std'
opt_var = 'var'


class ABCKernel(BaseDataProcess, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._weight = kwargs[opt_weight] if opt_weight in kwargs else 1

    def fit(self, mean, std):
        self._mean = mean
        self._std = std
        pass

    @abc.abstractmethod
    def map(self, x):
        return NotImplemented


class GeneralKernel(ABCKernel):
    """
    여러가지 kernel을 option에 따라서 선택할 수 있는 class입니다.

    kernelname 종류는 다음과 같습니다.

    - 'gaussian': gaussian kernel

    - 'epanechnikov': epanechnikov kernel
    """

    def __init__(self, kernelname, **kwargs):
        """

        :param kernelname: kernel name의 종류는 다음과 같습니다.

         - gaussian: gaussian kernel
         - epanechnikov: epanechnikov kernel

        :param weight:
        :return:
        """
        super().__init__(**kwargs)
        self._kernel = kernelname
        self.__k__ = self.__kernelselect__()

    def map(self, x):
        return self.__k__(x, self._mean, self._std)

    def __kernelselect__(self):
        """
        입력받은 kernel name에 맞는 kernel function을 return합니다.
        :return: target kernel function
        """

        if self._kernel == 'gaussian':
            return kernel_gaussian
        elif self._kernel == 'epanechnikov':
            return kernel_epanechnikov
        elif self._kernel == 'none':
            return lambda x, m, c: x
        else:
            raise AttributeError("kernel name does Not compitible.")


class KernelGaussian(ABCKernel):
    """
    gaussian kernel object
    """
    def map(self, x):
        return kernel_gaussian(x, self._mean, self._std)


class KernelEpanechnikov(ABCKernel):
    """
    epanechnikov kernel object
    """
    def map(self, x):
        return kernel_epanechnikov(x, self._mean, self._std)


##########################
# TODO 한 번 생각해보세요...
# 이 아래부턴 현재 작성된 class 안에 넣을 수 있습니다.
# 넣을건지 말건지는 니가 정하세요.
##########################
def formula_gaussian_weight(d, weight):
    return math.exp(-1 * d * weight)


def kernel_weight(size):
    """
    weight generator for kernel
    :param size: length of range
    :return: weight
    """
    mid = int(size / 2)
    for i in range(size):
        yield 2 ** (i - mid)


def distance_euclidean(x, m):
    """
    Euclidean distance를 계산합니다.
    :param x: array-like, 거리를 계산할 대상
    :param m: array-like, 기준점(평균)
    :return: float-like
    """
    # d = (sum((a - b) ** 2 for a, b in zip(x, m)) ** 0.5) / len(x)
    d = sum((a - b) ** 2 for a, b in zip(x, m)) ** 0.5
    return d


def formular_mahalanobis(val, mean, std):
    return 0 if val == mean or std == 0 else ((val - mean) ** 2) / (std ** 2)


def distance_mahalanobis(x, mean, std):
    """
    Mahalanobis distance를 계산합니다.
    :param x: array-like, 거리를 계산할 대상
    :param mean: array-like, 평균
    :param std: array-like, 표준 편차
    :return: float-like
    """
    d = sum(formular_mahalanobis(a, m, s) for a, m, s in zip(x, mean, std)) / len(x)
    d **= 0.5
    return d


def formula_gaussian(mahalanobis):
    return math.exp(-0.5 * (mahalanobis ** 2))
    # return 1 / ((2 * dataprocess.pi) ** 0.5) * dataprocess.exp(-0.5 * (mahalanobis ** 2))  # ORIGINAL


def kernel_gaussian(x, mean, std):
    _x = distance_mahalanobis(x, mean, std) if hasattr(x, '__iter__') else formular_mahalanobis(x, mean, std)
    __x = formula_gaussian(_x)
    return __x


def kernel_epanechnikov(iter_x, iter_m, iter_s):
    """
    Epanechnokiv Kernel Function

    :param iter_x:
    :param iter_m:
    :param iter_s:
    :return:
    """
    x = distance_mahalanobis(iter_x, iter_m, iter_s)
    x = 1 - x ** 2 if abs(x) <= 1 else 0.
    # x = 0.75 * (1 - x ** 2) if abs(x) <= 1 else 0.  # ORIGINAL
    return x
