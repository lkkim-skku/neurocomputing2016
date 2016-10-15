"""
model의 distribution과 관련된 클래스와 함수를 정의합니다.
beta, fld등이 있습니다. FLD는 아직...없습니다.
"""
# from scipy.stats import beta as scipybeta
from model.dataprocess import *
import controller
import time
import sys

opt_estimator = 'estimator'
param_mm = 'moment matching'
param_sf = 'sample fit'
param_ct = 'custom'
param_ms = 'moment searching'


class BetaEstimator:
    """
    Beta Model을 modeling합니다.
    option은 다음과 같습니다.

    - estimator: beta parameter를 estimate할 수 있는 알고리즘을 고릅니다.\n
    내장된 estimator는 문자열로 선택할 수 있으며, 개별적으로 만든 estimator는 다음의 형식을 따르면 됩니다.::
    def myestimator(*args): return alpha, beta

    개별적으로 만든 estimator는 다음의 예시로 적용할 수 있습니다.::
    BetaEstimator(estimator=myestimator)

    - alpha: beta parameter 중 alpha입니다.

    - beta: beta parameter 중 beta입니다.

    Oshiete! 먄약 alpha와 beta 모두 입력한다면 어떻게 해야할까요?
    """

    def __init__(self, estimator=param_mm):
        if hasattr(estimator, '__call__'):
            self._estimator = param_ct
            self.__estimator__ = estimator
            self._istimator = False
        else:
            self._estimator = estimator
            self.__estimator__ = BetaEstimator.__estimator_select__(self._estimator)
            self._istimator = True
        self._alpha, self._beta = 1, 1
        # self._data = []
        self._empirical = []
        self._betaobj = Betadist(1, 1)  # scipybeta  # TODO 나중에 combinedbeta를 만들 계획입니다.
        self._ksresult = 0

    @property
    def ap(self):
        return self._alpha

    @property
    def bt(self):
        return self._beta

    @property
    def beta(self):
        return self._betaobj

    @property
    def empirical(self):
        return self._empirical

    def parameter(self):
        return self._alpha, self._beta

    def fit(self, data: BaseData):
        """
        beta parameter를 fit합니다.
        fitting은 다음의 과정을 거칩니다.

        #. rv의 histogram을 구합니다.
        #. histogram of rv의 CDF를 구합니다.
        #. histogram CDF of rv의 각 bin이 나타내는 Frequency(histogram of rv의 y축 값)를 구합니다.
        이것은 rv frequency가 됩니다.
          - 논문 상에서는 histogram of rv가 아니라 CDF of rv입니다. 하지만 CDF of rv는
          KS-test에 부적합할 수 있습니다. 너무 많기 때문이죠. 따라서 이를 피하기 위해 우리는 100개로 한정합니다.
          100개로 한정하려면 어떻게 하는 게 좋을까요? 간단합니다. histogram의 bins를 100개로 고정하면 됩니다.
        #. beta의 parameter를 계산합니다. 이는 initial parameter입니다.
        #. initial parameter를 beta function이 rv frequency를 잘 표현할 수 있도록 fitting합니다.
        #. KS-test 수행 결과 중 하나인 p-value가 0.05 이상이면 accept, 이하면 다시 fitting합니다.

        :param BaseData data: fitting할 대상의 data입니다.
        :return: :class:`BetaEstimator`
        """

        self._empirical = histo_cudif(data.data, 100)
        ap, bt = self.__estimator__(data)

        betaobj = Betadist(ap, bt)
        d, p = ksone(self._empirical, betaobj.cdf)  # , N=len(self._empirical))

        self._alpha, self._beta = ap, bt
        self._betaobj = betaobj
        self._ksresult = p
        # controller.plot_lines(str(time.time()), str(time.time()),
        #                       {'name': 'ecdf', 'y': self._empirical, 'x': [x * 0.01 for x in range(100)]},
        #                       {'name': 'bcdf', 'y': [self.predict(x * 0.01) for x in range(100)],
        #                        'x': [x * 0.01 for x in range(100)]}
        #                       )
        return self

    @property
    def ksresult(self):
        return self._ksresult

    def predict(self, ranvar):
        """
        kernelized random variable을 입력 받으면, betaCDF에 대입했을 때의 출력을 반환합니다.
        :param ranvar:
        :return:
        """
        # if ranvar >= 0.99:  # TODO 이거 나을텍용인데 진짜 꼭 고쳐라
        #     return 1.  # 꼭꼭 무조건 고처라
        betaoutput = self._betaobj.cdf(ranvar)
        return betaoutput

    @staticmethod
    def __estimator_select__(estimator_name=param_mm):
        """
        내장된 estimator를 선택합니다. 다음은 estimator 목록입니다.

        - :func:`be_momentmatch` : moment matching

        - :func:`besamplefitting` : Fitting Beta Distribution Based on Sample Data에서 제안한
        beta fitting 기법입니다. 최초 parameter는 :func:`be_momentmatch` 으로 예측합니다.

        :param str estimator_name: 내장된 estimator를 나타내는 이름
        :return: function address
        """
        if estimator_name in param_sf:
            return be_samplefitting
        else:
            return be_momentmatch

    pass


def be_momentmatch(data: BaseData):
    """
    approximate beta parameters with mean and variance

    :param BaseData data: sample data
    :return:
    """
    mean, var = data.mean, data.var
    lower, upper = 0., 1.  # feature scaling한 뒤니까 당연히 0과 1이죠.
    ml = mean - lower
    um = upper - mean
    ap = (ml / (upper - lower)) * (((ml * um) / var) - 1)
    bt = ap * (um / ml)

    return ap, bt


def be_samplefitting(data: BaseData):
    """
    Fitting Beta Distributions Based on Sample Data

    :param BaseData data: sample data
    :return:
    """
    alpha, beta = be_momentmatch(data)
    # alpha, beta, _, _= scipybeta.fit(data.data, alpha, beta)  # 성능이 쓰레기
    # TODO 여기서부터 본격적인 samplefitting입니다.

    return alpha, beta


"""
Numerical recipes The art of Scientific Computing, 3rd edition, Cambridge University Press
"""


def gammln(xx: float):
    """

    :param float xx:
    :return: the value ln(gamma(xx)) for xx > 0.
    :reference: Numerical recipes The art of Scientific Computing,
    3rd edition, Cambridge University Press, pp.257
    """
    cof = [57.1562356658629235, -59.5979603554754912, 14.1360979747417471, 
           -0.491913816097620199, .339946499848118887e-4, .465236289270485756e-4, 
           -.983744753048795646e-4, .158088703224912494e-3, -.210264441724104883e-3, 
           .217439618115212643e-3, -.164318106536763890e-3, .844182239838527433e-4, 
           -.261908384015814087e-4, .368991826595316234e-5]
    if xx <= 0:
        raise ValueError("bad arg in gammln")
    _x = xx
    y = _x
    tmp = _x + 5.24218750000000000  # Rational 671/128
    tmp = (_x + .5) * math.log(tmp) - tmp
    ser = 0.999999999999997092
    for _j in range(14):
        y += 1
        ser += cof[_j] / y
    return tmp + math.log(2.5066282746310005 * ser / _x)


class Gauleg18:
    """
    Abscissas and weights for Gauss-Legendre quadrature.
    :reference: Numerical recipes The art of Scientific Computing,
    3rd edition, Cambridge University Press, pp.262-3
    """
    ngau = 18
    y = [0.0021695375159141994, 0.011413521097787704, 0.027972308950302116,
         0.051727015600492421, 0.082502225484340941, 0.12007019910960293,
         0.16415283300752470, 0.21442376986779355, 0.27051082840644336,
         0.33199876341447887, 0.39843234186401943, 0.46931971407375483,
         0.54413605556657973, 0.62232745288031077, 0.70331500465597174,
         0.78649910768313447, 0.87126389619061517, 0.95698180152629142]
    w = [0.0055657196642445571, 0.012915947284065419, 0.020181515297735382,
         0.027298621498568734, 0.034213810770299537, 0.040875750923643261,
         0.047235083490265582, 0.053244713977759692, 0.058860144245324798,
         0.064039797355015485, 0.068745323835736408, 0.072941885005653087,
         0.076598410645870640, 0.079687828912071670, 0.082187266704339706,
         0.084078218979661945, 0.085346685739338721, 0.085983275670394821]


class Gamma(Gauleg18):
    """
    Object for incomplete gamma function. Gauleg18 provides coefficients for Gauss-Legendre quadrature.

    :reference: Numerical recipes The art of Scientific Computing,
    3rd edition, Cambridge University Press, pp.261-2
    """
    ASWITCH = 100
    EPS = sys.float_info.epsilon
    FPMIN = sys.float_info.min / sys.float_info.epsilon
    gln = 0.

    @staticmethod
    def gammpapprox(a: float, x: float, psig: int):
        """
        Incomplete gamma by quadrature.

        :param a:
        :param x:
        :param psig:
        :return: P(a, x) or Q(a, x), when psig is 1 or 0, respectively.
        User should not call directly.

        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.262
        """
        a1 = a - 1
        lna1, sqrta1 = math.log(a1), a1 ** .5
        gln = gammln(a)
        # Set how far to integrate into the tail:
        if x > a1:
            xu = max(a1 + 11.5 * sqrta1, x + 6. * sqrta1)
        else:
            xu = max(0., min(a1 - 7.5 * sqrta1, x - 5. * sqrta1))
        _sum = 0
        for j in range(Gamma.ngau):  # Gauss-Legendre.
            t = x + (xu - x) * Gamma.y[j]
            _sum += Gamma.w[j] * math.exp(-(t - a1) + a1 * (math.log(t) - lna1))

        ans = _sum * (xu - x) * math.exp(a1 * (lna1 - 1.) - gln)

        return 1. - ans if ans > 0 else -ans if psig else ans if ans >= 0 else 1. + ans

    @staticmethod
    def gser(a: float, x: float):
        """

        :param a:
        :param x:
        :return: The incomplete gamma function P(a, x) evaluated by its series representation.
        Also sets ln(gamma(a) as :func:`gammaln`. User should not call directly.

        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.261
        """
        gln = gammln(a)
        ap = a
        _sum = 1. / a
        _del = _sum
        while True:
            ap += 1
            _del *= x / ap
            _sum += _del
            if abs(_del) < abs(_sum) * Gamma.EPS:
                return _sum * math.exp(-x + a * math.log(x) - gln)

    @staticmethod
    def gammp(a: float, x: float):
        """
        :param float a:
        :param float x:
        :return: the incomplete gamma function P(a, x).
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.261
        """
        if x < 0. or a <= 0.:
            raise ValueError("bad args in gammp")
        if x == 0.:
            return 0.
        elif a >= Gamma.ASWITCH:
            return Gamma.gammpapprox(a, x, 1)  # Quadrature.
        elif x < a + 1.:
            return Gamma.gser(a, x)  # Use the series represntation.
        else:
            return 1. - Gamma.gcf(a, x)  # Use the continued fraction represntation.

    @staticmethod
    def gammq(a: float, x: float):
        """
        :param float a:
        :param float x:
        :return: the incomplete gamma function Q(a, x) = 1 - P(a, x).
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.261
        """
        if x < 0. or a <= 0.:
            raise ValueError("bad args in gammq")
        if x == 0.:
            return 1.
        elif a >= Gamma.ASWITCH:
            return Gamma.gammpapprox(a, x, 0)  # Quadrature.
        elif x < a + 1.:
            return 1. - Gamma.gser(a, x)  # Use the series represntation.
        else:
            return Gamma.gcf(a, x)  # Use the continued fraction represntation.

    @staticmethod
    def gcf(a: float, x: float):
        """

        :param a:
        :param x:
        :return: The incomplete gamma function Q(a, x) evaluated by its continued fraction representation.
        Also sets ln(gamma(a)) as :func:`gammaln` . User should not call directly.

        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.261-262
        """
        gln = gammln(a)
        b, c = x + 1. - a, 1. / Gamma.FPMIN  # Set up for evaluating continued fraction\
        d = 1. / b  # by modified Lentz's method with b_0 = 0.
        h, i = d, 0

        while True:  # Iterate to convergence.
            an = -i * (i - a)
            b += 2.
            d = an * d + b
            if abs(d) < Gamma.FPMIN:
                d = Gamma.FPMIN
            c = b + an / c

            if abs(c) < Gamma.FPMIN:
                c = Gamma.FPMIN
            d = 1. / d
            _del = d * c
            h *= _del
            if abs(_del - 1.) <= Gamma.EPS:
                break
            i += 1

        return math.exp(-x + a * math.log(x) - gln) * h  # Put factors in front.

    @staticmethod
    def invgammp(p: float, a: float):
        """
        :param p:
        :param a:
        :return: x such that P(a, x) = p for an argument p between 0 and 1.
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.263
        """
        eps = 1.e-8
        a1 = a - 1
        gln = gammln(a)

        if a <= 0.:
            raise ValueError("a must be pos in invgammap")
        if p >= 1.:
            return max(100., a + 100. * a ** .5)
        if p <= 0.:
            return 0.0
        afac, lna1 = 0., 0.
        if a > 1.:
            lna1 = math.log(a1)
            afac = math.exp(a1 * (lna1 - 1.) - Gamma.gln)
            pp = p if p < .5 else 1. - p
            t = (-2. * math.log(pp)) ** .5
            xx = (2.30753 + t * 0.27061) / (1. + t * (0.99229 + t * 0.04481)) - t
            if p < 0.5:
                xx = -xx
            xx = max(1e-3, a * pow(1. - 1. / (9. * a) - xx / (3. * a ** .5), 3))
        else:
            t = 1.0 - a * (0.253 + a * 0.12)
            if p < t:
                xx = pow(p / t, 1. / a)
            else:
                xx = 1. - math.log(1. - (p - t) / (1. - t))

        for j in range(12):
            if xx <= 0.0:
                return 0.0
            err = Gamma.gammp(a, xx) - p
            if a > 1.:
                t = afac * math.exp(-(xx - a1) + a1 * (math.log(xx) - lna1))
            else:
                t = math.exp(-xx + a1 * math.log(xx) - gln)
            u = err / t
            t = u / (1. - .5 * min(1., u * ((a - 1.) / xx - 1)))
            xx -= t
            if xx <= 0.:
                xx = .5 * (xx + t)
            if abs(t) < eps * xx:
                break

        return xx


class Beta(Gauleg18):
    """
    Object for incomplete beta function. Gauleg18 provides coefficients for Gauss-Legendre quadrature.

    :reference: Numerical recipes The art of Scientific Computing,
    3rd edition, Cambridge University Press, pp.272-273
    """
    SWITCH = 3000  # When to switch to quadrature method.
    EPS = sys.float_info.epsilon
    FPMIN = sys.float_info.min / sys.float_info.epsilon

    @staticmethod
    def betai(a: float, b: float, x: float):
        """

        :param a:
        :param b:
        :param x:
        :return: Incomplete beta function I_x(a, b) for positive a and b, and x between 0 and 1.
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.272
        """
        if a <= 0.0 or b <= 0.0:
            raise ValueError("Bad a or b in routine betai")
        if x < 0.0 or x > 1.0:
            raise ValueError("Bad x in routine betai")
        if x == 0.0 or x == 1.0:
            return x
        if a > Beta.SWITCH and b > Beta.SWITCH:
            return Beta.__betaiapprox__(a, b, x)
        bt = math.exp(gammln(a + b) - gammln(a) - gammln(b) + a * math.log(x) + b * math.log(1. - x))
        if x < (a + 1.) / (a + b + 2.):
            return bt * Beta.__betacf__(a, b, x) / a
        else:
            return 1. - bt * Beta.__betacf__(b, a, 1. - x) / b

    @staticmethod
    def __betacf__(a: float, b: float, x: float):
        """
        Evaluates continued fraction for incomplete beta function by modified Lentz’s method(5.2).
        User should not call directly.

        :param a:
        :param b:
        :param x:
        :return:
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.272
        """
        # There q's will be used in factors that occur in the coefficients(6.4.6).
        qab, qap, qam = a + b, a + 1., a - 1.
        c = 1.  # First step of Lentz's method.
        d = 1. - qab * x / qap
        if abs(d) < Beta.FPMIN:
            d = Beta.FPMIN
        d = 1. / d
        h = d
        for m in range(1, 10000):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1. + aa * d  # One step ( the even one) of the recurrence.
            if abs(d) < Beta.FPMIN:
                d = Beta.FPMIN
            c = 1. + aa / c
            if abs(c) < Beta.FPMIN:
                c = Beta.FPMIN
            d = 1. / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1. + aa * d  # Next step of the recurrence ( the odd one).
            if abs(d) < Beta.FPMIN:
                d = Beta.FPMIN
            c = 1. + aa / c
            if abs(c) < Beta.FPMIN:
                c = Beta.FPMIN
            d = 1. / d
            _del = d * c
            h *= _del
            if abs(_del - 1.) <= Beta.EPS:  # Are we done?
                break

        return h

    @staticmethod
    def __betaiapprox__(a: float, b: float, x: float):
        """
        Incomplete beta by quadrature. User should not call directly.
        :param a:
        :param b:
        :param x:
        :return: I_x(a, b).
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.272-273
        """
        a1, b1, mu = a - 1., b - 1., a / (a + b)
        lnmu, lnmuc = math.log(mu), math.log(1. - mu)
        t = (a * b / (((a + b) ** 2) * (a + b + 1.))) ** .5
        if x > a / (a + b):  # Set how far to integrate into the tail:
            if x >= 1.:
                return 1.
            xu = min(1., max(mu + 10. * t, x + 5. * t))
        else:
            if x <= 0.:
                return 0.
            xu = max(0., min(mu - 10. * t, x - 5. * t))

        _sum = 0
        for j in range(18):  # Gauss-Legendre.
            t = x + (xu - x) * Beta.y[j]
            _sum += Beta.w[j] * math.exp(a1 * (math.log(t) - lnmu) + b1 * (math.log(1 - t) - lnmuc))

        ans = _sum * (xu - x) * math.exp(a1 * lnmu - gammln(a) + b1 * lnmuc - gammln(b) + gammln(a + b))
        return 1.0 - ans if ans > 0. else -ans

    @staticmethod
    def invbetai(p: float, a: float, b: float):
        """
        Inverse of incomplete beta function.
        :param p:
        :param a:
        :param b:
        :return:  x such that I_x(a, b) = p for argument p between 0 and 1.
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.273
        """
        eps = 1.e-8
        a1, b1 = a - 1., b - 1.
        if p <= 0.:
            return 0.
        elif p >= 1.:
            return 1.
        elif a >= 1. and b >= 1.:  # Set initial guess. See text.
            pp = p if p < 0.5 else 1. - p
            t = (-2. * math.log(pp)) ** .5
            x = (2.30753 + t * 0.27061) / (1. + t * (0.99229 + t * 0.04481)) - t
            if p < 0.5:
                x = -x
            al = ((x ** 2) - 3.) / 6.
            h = 2. / (1. / (2. * a - 1.) + 1. / (2. * b - 1.))
            w = (x * ((al + h) ** .5) / h) - (1. / (2. * b - 1) - 1. / (2. * a - 1.)) * (al + 5. / 6. - 2. / (3. * h))
            x = a / (a + b * math.exp(2. * w))
        else:
            lna, lnb = math.log(a / (a + b)), math.log(b / (a + b))
            t = math.exp(a * lna) / a
            u = math.exp(b * lnb) / b
            w = t + u
            if p < t / w:
                x = pow(a * w * p, 1. / a)
            else:
                x = 1. - pow(b * w * (1. - p), 1. / b)

        afac = -gammln(a) - gammln(b) + gammln(a + b)
        for j in range(10):
            if x == 0. or x == 1.:  # a or b too small for accurate calculation.
                return x
            err = Beta.betai(a, b, x) - p
            t = math.exp(a1 * math.log(x) + b1 * math.log(1. - x) + afac)
            u = err / t  # Halley:
            t = u / (1. - .5 * min(1., u * (a1 / x - b1 / (1. - x))))
            x -= t
            if x <= 0.:  # Bisect if x tries to go neg or > 1.
                x = .5 * (x + t)
            if x >= 1.:
                x = .5 * (x + t + 1.)
            if abs(t) < eps * x and j > 0:
                break

        return x


class Betadist(Beta):
    """
    Beta distribution, derived from the :class:`Beta` .

    :reference: Numerical recipes The art of Scientific Computing,
    3rd edition, Cambridge University Press, pp.334
    """
    def __init__(self, alph, bet):
        """
        initialize a and b.
        :param alph:
        :param bet:
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.334
        """
        if alph <= 0. or bet <= 0.:
            raise ValueError("bad alph,bet in Betadist")
        self._fac = gammln(alph + bet) - gammln(alph) - gammln(bet)
        self._alph, self._bet = alph, bet

    def p(self, x: float):
        """
        :param x:
        :return: probability density function.
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.334
        """
        if x <= 0. or x >= 1.:
            raise ValueError("bad x in Betadist")
        return math.exp((self._alph - 1.) * math.log(x) + (self._bet - 1.) * math.log(1. - x) + self._fac)

    def cdf(self, x: float):
        """
        0보다 작으면 0, 1보다 크면 1로 계산합니다.
        :param x:
        :return: cumulative distribution function.
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.334
        """
        # ORIGINAL CODE START
        # if x < 0. or x > 1.:
        #     raise ValueError("bad x in Betadist")
        # return Betadist.betai(self._alph, self._bet, x)
        # ORIGINAL CODE END
        return Betadist.betai(self._alph, self._bet, 0. if x < 0. else (1. if x > 1. else x))

    def invcdf(self, p: float):
        """
        160508: 0보다 작으면 0, 1보다 크면 1로 수정했습니다.
        :param p:
        :return: inverse cumulative distribution function.
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.334
        """
        # if p < 0. or p > 1.:
        #     raise ValueError("bad p in Betadist")
        if p < 0:
            return Betadist.invbetai(0., self._alph, self._bet)
        elif p > 1:
            return Betadist.invbetai(1., self._alph, self._bet)
        return Betadist.invbetai(p, self._alph, self._bet)

    @property
    def alph(self):
        return self._alph

    @property
    def bet(self):
        return self._bet


def invxlogx(y: float):
    """
    For negative y, 0 > y > -e ** -1, return x such that y = x * math.log(x).
    The value returned is always the smaller of the two roots and is in the range 0 < x < e ** -1.

    :param y:
    :return:
    :reference: Numerical recipes The art of Scientific Computing,
    3rd edition, Cambridge University Press, pp.309
    """
    ooe = 0.367879441171442322
    t, u, to = 0., 0., 0.

    if y >= 0. or y <= -ooe:
        raise ValueError("no such inverse value")
    if y < -0.2:
        # First approximation by inverse of Taylor series.
        u = math.log(ooe - (2. * ooe * (y + ooe)) ** 0.5)
    else:
        u = -10.

    loop_condition = 1.
    while loop_condition > 1e-15:
        t = (math.log(y / u) - u) * (u / (1. + u))
        u += t
        if t < 1e-8 and abs(t + to) < 0.01 * abs(t):
            break
        to = t
        loop_condition = abs(t / u)
    return math.exp(u)


class KSdist:
    """
    Kolmogorov-Smirnov cumulative distribution functions and their inverses.

    :reference: Numerical recipes The art of Scientific Computing,
    3rd edition, Cambridge University Press, pp.335-336

    """

    @staticmethod
    def pks(z: float):
        """
        :param z: Kolmogorov-Smirnov random variable
        :return: cumulative distribution function.
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.335
        """
        if z < 0.:
            raise ValueError("bad z in KSdist")
        if z == 0.:
            return 0.
        if z < 1.18:
            y = math.exp(-1.23370055013616983 / (z ** 2))
            return 2.25675833419102515 * (-math.log(y)) ** 0.5 * (y + pow(y, 9) + pow(y, 25) + pow(y, 49))
        else:
            x = math.exp(-2. * z ** 2)
            return 1. - 2. * (x - pow(x, 4) + pow(x, 9))

    @staticmethod
    def qks(z: float):
        """
        :param float z: Kolmogorov-Smirnov random variable
        :return: complementary cumulative distribution function.
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.335
        """
        if z < 0.:
            raise ValueError("bad z in KSdist")
        if z == 0.:
            return 1.
        if z < 1.18:
            return 1. - KSdist.pks(z)
        xx = math.exp(-2. * z ** 2)

        return 2. * (xx - pow(xx, 4) + pow(xx, 9))

    @staticmethod
    def invqks(q: float):
        """
        :param q: complementary cumulative distribution function.
        :return: inverse of the complementary cumulative distribution function.
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.335-336
        """
        if q <= 0. or q > 1.:
            raise ValueError("bad q in KSdist")
        if q == 1.:
            return 0.
        if q > 0.3:
            f = -0.392699081698724155 * (1. - q) ** 2
            y = invxlogx(f)  # Initial guess.

            loopcondition = 1
            while loopcondition > 1e-15:
                yp = y
                logy = math.log(y)
                ff = f / (1. + pow(y, 4) + pow(y, 12)) ** 2
                u = (y * logy - ff) / (1. + logy)  # Newton's method correction
                t = u / max(0.5, 1. - 0.5 * u / (y * (1. + logy)))  # Hally.
                y = yp - t  # 원문에는 y = y - t인데, 이걸 이렇게 쓸만한 책이 아니라서 다르게 해석함.
                loopcondition = abs(t / y)

            return 1.57079632679489662 / (-math.log(y)) ** 0.5
        else:
            xx = 0.03
            loopcondition = 1
            while loopcondition > 1e-15:  # Iteration(6.14.59).
                xp = xx
                xx = 0.5 * q + pow(xx, 4) - pow(xx, 9)
                if xx > 0.06:
                    xx += pow(xx, 16) - pow(xx, 25)
                loopcondition = abs((xp - xx) / xx) > 1e-15
            return (-0.5 * math.log(xx)) ** 0.5

    @staticmethod
    def invpks(p: float):
        """
        :param p: cumulative distribution function.
        :return: inverse of the cumulative distribution function.
        :reference: Numerical recipes The art of Scientific Computing,
        3rd edition, Cambridge University Press, pp.336
        """
        return KSdist.invqks(1. - p)


def ksone(data: list, func: callable):
    """
    이 함수의 원본은 사용하기에 부적합합니다.
    우리가 비교해야할 대상은 data의 x와 func(y)입니다.
    이 때 y는 data의 index가 되어야 합니다.
    그런데 이 함수는 data의 y와 func(x)를 비교합니다.
    따라서 이 부분을 위의 조건대로 수정하였습니다.
    :author: Leesuk Kim, lktime@skku.edu

    Given an array data[0..n-1], and given a user-supplied function of a single variable func that is
    a cumulative distribution function ranging from 0 (for smallest values of its argument) to 1 (for
    largest values of its argument), this routine returns the K–S statistic d and the p-value prob.
    Small values of prob show that the cumulative distribution function of data is significantly
    different from func. The array data is modified by being sorted into ascending order.

    :param list data: array of cumulative distribution function
    :param callable func: cumulative distribution function
    :return d: KS statistics
    :return prob: p-value
    :reference: Numerical recipes The art of Scientific Computing,
    3rd edition, Cambridge University Press, pp.737-738
    """
    n = len(data)
    step = 1 / n
    fo = 0.
    sorted(data)
    # If the data are already sorted int ascending order ,then this call can be omitted.
    en = n ** .5
    d = 0
    for j in range(n):  # Loop over the sorted data point.
        fn = data[j]  # Data's CDF after this step.
        ff = func(j * step)  # Compare to the user-supplied function.
        # fn = (j + 1) / en  # Data's CDF after this step.
        # ff = func(data[j])  # Compare to the user-supplied function.
        dt = max(abs(fo - ff), abs(fn - ff))  # Maximum distance.
        if dt > d:
            d = dt
        fo = fn

    prob = KSdist.qks((en + 0.12 + 0.11 / en) * d)

    return d, prob


def kstwo(data1: list, data2: list):
    """
    Given an array data1[0..n1-1], and an array data2[0..n2-1], this routine returns the K–S
    statistic d and the p-value prob for the null hypothesis that the data sets are drawn from the
    same distribution. Small values of prob show that the cumulative distribution function of data1
    is significantly different from that of data2. The arrays data1 and data2 are modified by being
    sorted into ascending order.

    :param list data1: array of cumulative distribution function
    :param list data2: array of cumulative distribution function
    :return d: KS statistics
    :return prob: p-value
    :reference: Numerical recipes The art of Scientific Computing,
    3rd edition, Cambridge University Press, pp.738
    """
    j1, j2, n1, n2 = 0, 0, len(data1), len(data2)
    fn1, fn2 = 0., 0.
    sorted(data1), sorted(data2)
    en1, en2 = n1, n2
    d, prob = 0, 0

    while j1 < n1 and j2 < n2:  # If we are not done
        d1, d2 = data1[j1], data2[j2]
        if d1 < d2:  # Next step is in data1.
            loopcondition = True
            while loopcondition:
                j1 += 1
                fn1 = j1 / en1
                loopcondition = j1 < n1 and d1 == data1[j1]
        if d2 < d1:  # Next step is in data2.
            loopcondition = True
            while loopcondition:
                j2 += 1
                fn2 = j2 / en2
                loopcondition = j2 < n2 and d2 == data2[j2]
        dt = abs(fn2 - fn1)
        if dt > d:
            d = dt

        en = (en1 * en2 / (en1 + en2)) ** 0.5
        prob = KSdist.qks((en + 0.12 + 0.11 / en) * d)  # Compute p-value <- 0.23이 아니라 0.11/en임!

    return d, prob
