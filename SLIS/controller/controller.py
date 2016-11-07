"""
수학적이지 않고 프로그래밍에서만 필요한 함수들을 정의합니다.
"""
import matplotlib.pyplot as plt
import random


def mymax(itervar, **kwargs):
    """
    max인데, max의 결과가 일정 기준 이하면 default를 리턴
    :param dict itervar:
    :param kwargs:
    :return:
    """
    if 'key' in kwargs:
        mval = max(itervar, key=kwargs['key'])
    else:
        mval = max(itervar)
    default = kwargs['default'] if 'default' in kwargs else ValueError
    if 'underbound' in kwargs:
        mval = default if itervar[mval] < kwargs['underbound'] else mval
    return mval


def mymin(itervar, **kwargs):
    """
    max인데, max의 결과가 일정 기준 이하면 default를 리턴
    :param dict itervar:
    :param kwargs:
    :return:
    """
    if 'key' in kwargs:
        mval = min(itervar, key=kwargs['key'])
    else:
        mval = min(itervar)
    default = kwargs['default'] if 'default' in kwargs else ValueError
    if 'underbound' in kwargs:
        mval = default if itervar[mval] > kwargs['upperbound'] else mval
    return mval


def move_keyvalue(srcdict: dict, desdict: dict, key):
    """
    기존 dict의 key와 value을 목표 dict에 추가하고 기존 dict의 key와 value를 삭제합니다.
    어떤 의미에서는 cut and paste라고 볼 수 있습니다.

    :param dict srcdict: source dictionary
    :param dict desdict: destination dictionary
    :param key: key for source and destination dictionary
    :return: Nothing
    """
    if key in srcdict:
        desdict[key] = srcdict.pop(key)
    return


def classionary(content, target):
    """
    content와 target으로 나눠진 learning data를 target에 따라 묶은 dict로 만들어준다.

    :param list content: array-like. it contains content for learning
    :param list target: array-like. it contains target for learning
    :return dict:
    """
    kv = {}

    for c, t in zip(content, target):
        if t in kv:
            kv[t].append(c)
        else:
            kv[t] = [c]

    return kv


def folding_160311(content, target):
    """
    16년 03월 11일에 작성한 folding algorithm

    - 클래스별로 나눠서 학습시킨다.

    - unknown을 정해서 일부 class를 unknown으로 설정하며 fold한다.

    - 0%, 10%, 20%, 30%, 40%, 50%로 나눈다.\n
      ex. 10%: unknown: 10, 20, 30, 40, 50 나머지는 known

    - 그런데 빡세게(?) 코딩했는데 애초에 python의 for문은 yield되서 순서가 무작워...ㄷㄷ
      따라서 크게 의미는 없음...ㅋㅋ

    :param list content: array-like. it contains content for learning
    :param list target: array-like. it contains target for learning
    :return:
        content for learing,
        target for learning,
        content for examinating,
        target for examinating
    """
    size = 6
    csize = size - 1
    lcon, ltar = [[] for _ in range(size)], [[] for _ in range(size)]
    econ, etar = [[] for _ in range(size)], [[] for _ in range(size)]

    learningmatters = classionary(content, target)

    for i, k in enumerate(learningmatters):
        modi = i % 10
        if modi >= csize:
            for _i in range(modi - csize + 1):
                for matters in learningmatters[k]:
                    ltar[_i].append(k), lcon[_i].append(matters)
        else:
            for _i in range(size):
                for matters in learningmatters[k]:
                    ltar[_i].append(k), lcon[_i].append(matters)
        # testing은 모든 class의 data를 다 넣습니다.
        for _i in range(size):
            for matters in learningmatters[k]:
                etar[_i].append(k), econ[_i].append(matters)

    return lcon, ltar, econ, etar


def detect_160323(p, size: int):
    """
    T-statistic
    :param p:
    :param size:
    :return:
    """
    # sigrange = .675 * (1 / (12 * size)) ** .5  # 0.5
    # sigrange = 1.645 * (1 / (12 * size)) ** .5   # 0.1
    sigrange = 1.96 * (1 / (12 * size)) ** .5  # 0.05
    # sigrange = 2.58 * (1 / (12 * size)) ** .5  # 0.01
    # sigrange = 2.81 * (1 / (12 * size)) ** .5  # 0.005
    # sigrange = 3.29 * (1 / (12 * size)) ** .5  # 0.001
    return True if .5 - sigrange < p < .5 + sigrange else False


def detect_160524(seqpval, mean_p, size: int):
    is_spm = detect_160323(mean_p, size)
    is_spv = seqpval > .05
    return is_spm or is_spv


def folding_160311_half(content, target):
    """
    16년 03월 11일에 작성한 folding algorithm

    - 클래스별로 나눠서 학습시킨다.

    - unknown을 정해서 일부 class를 unknown으로 설정하며 fold한다.

    - 0%, 10%, 20%, 30%, 40%, 50%로 나눈다.\n
      ex. 10%: unknown: 10, 20, 30, 40, 50 나머지는 known

    - 그런데 빡세게(?) 코딩했는데 애초에 python의 for문은 yield되서 순서가 무작워...ㄷㄷ
      따라서 크게 의미는 없음...ㅋㅋ

    :param list content: array-like. it contains content for learning
    :param list target: array-like. it contains target for learning
    :return:
        content for learing,
        target for learning,
        content for examinating,
        target for examinating
    """
    size = 6
    csize = size - 1
    lcon, ltar = [[] for _ in range(size)], [[] for _ in range(size)]
    econ, etar = [[] for _ in range(size)], [[] for _ in range(size)]

    learningmatters = classionary(content, target)

    for i, k in enumerate(learningmatters):
        modi = i % 10
        lm_half = int(len(learningmatters[k]) / 2)
        if modi >= csize:
            for _i in range(modi - csize + 1):
                for matters in learningmatters[k][lm_half:]:
                    ltar[_i].append(k), lcon[_i].append(matters)
        else:
            for _i in range(size):
                for matters in learningmatters[k][lm_half:]:
                    ltar[_i].append(k), lcon[_i].append(matters)

        # testing은 모든 class의 data를 다 넣습니다.
        for _i in range(size):
            for matters in learningmatters[k][:lm_half]:
                etar[_i].append(k), econ[_i].append(matters)

    return lcon, ltar, econ, etar


def folding_160323(content, target, args=()):
    """
    지정한 class만 unknown으로 만듭니다.

    - 0%, 10%, 20%, 30%, 40%, 50%로 나눈다.\n
      ex. 10%: unknown: 10, 20, 30, 40, 50 나머지는 known

    - 그런데 빡세게(?) 코딩했는데 애초에 python의 for문은 yield되서 순서가 무작워...ㄷㄷ
      따라서 크게 의미는 없음...ㅋㅋ

    :param list content: array-like. it contains content for learning
    :param list target: array-like. it contains target for learning
    :param tuple args: array-like. 특정 클래스 이름
    :return:
        content for learing,
        target for learning,
        content for examinating,
        target for examinating
    """
    size = 1
    lcon, ltar = [[] for _ in range(size)], [[] for _ in range(size)]
    econ, etar = [[] for _ in range(size)], [[] for _ in range(size)]

    learningmatters = classionary(content, target)

    for i, k in enumerate(learningmatters):
        if k not in args:
            for matters in learningmatters[k]:
                ltar[0].append(k), lcon[0].append(matters)
        # testing은 모든 class의 data를 다 넣습니다.
        for _i in range(size):
            for matters in learningmatters[k]:
                etar[0].append(k), econ[0].append(matters)

    return lcon, ltar, econ, etar


def folding_160323_half(content, target, args=()):
    """
    지정한 class만 unknown으로 만듭니다.

    - 0%, 10%, 20%, 30%, 40%, 50%로 나눈다.\n
      ex. 10%: unknown: 10, 20, 30, 40, 50 나머지는 known

    - 그런데 빡세게(?) 코딩했는데 애초에 python의 for문은 yield되서 순서가 무작워...ㄷㄷ
      따라서 크게 의미는 없음...ㅋㅋ

    :param list content: array-like. it contains content for learning
    :param list target: array-like. it contains target for learning
    :param tuple args: array-like. 특정 클래스 이름
    :return:
        content for learing,
        target for learning,
        content for examinating,
        target for examinating
    """
    size = 1
    lcon, ltar = [[] for _ in range(size)], [[] for _ in range(size)]
    econ, etar = [[] for _ in range(size)], [[] for _ in range(size)]

    learningmatters = classionary(content, target)

    for i, k in enumerate(learningmatters):
        lm_half = int(len(learningmatters[k]) / 2)
        if len([x for x in args if x in k]) == 0:
            for matters in learningmatters[k][lm_half:]:
                ltar[0].append(k), lcon[0].append(matters)
        for _i in range(size):
            for matters in learningmatters[k][lm_half:]:
                etar[0].append(k), econ[0].append(matters)

    return lcon, ltar, econ, etar


def folding_160411_half(content, target, args=()):
    """
    지정한 class만 unknown으로 만듭니다.
    그리고 5개의 fold는 test data의 갯수만 달라집니다.
    10, 20, 30, 40, 50개로 나눠집니다.

    - 그런데 빡세게(?) 코딩했는데 애초에 python의 for문은 yield되서 순서가 무작워...ㄷㄷ
      따라서 크게 의미는 없음...ㅋㅋ

    :param list content: array-like. it contains content for learning
    :param list target: array-like. it contains target for learning
    :param tuple args: array-like. 특정 클래스 이름
    :return:
        content for learing,
        target for learning,
        content for examinating,
        target for examinating
    """
    size = 5
    lcon, ltar = [[] for _ in range(size)], [[] for _ in range(size)]
    econ, etar = [[] for _ in range(size)], [[] for _ in range(size)]

    learningmatters = classionary(content, target)

    for i, k in enumerate(learningmatters):
        lm_half = int(len(learningmatters[k]) / 2)
        overten = len(learningmatters[k]) * .1

        if len([x for x in args if x in k]) == 0:
            for matters in learningmatters[k][lm_half:]:
                for x in range(size):
                    ltar[x].append(k), lcon[x].append(matters)
        for j in range(size):
            # for matters in pickme(learningmatters[k][:lm_half], int(overten)):
            for matters in learningmatters[k][:int(overten) * (j + 1)]:
                etar[j].append(k), econ[j].append(matters)

    return lcon, ltar, econ, etar


def folding_160624_half(content, target):
    """
    folding_160311_half를 다시 만들었습니다.
    :param content:
    :param target:
    :param args:
    :return:
    """
    size = 6
    csize = size - 2
    lcon, ltar = [[] for _ in range(size)], [[] for _ in range(size)]
    econ, etar = [[] for _ in range(size)], [[] for _ in range(size)]
    learningmatters = classionary(content, target)

    _log = []
    for i, k in enumerate(learningmatters):
        modi = i % 10
        _log.append(k)
        lm_half = int(len(learningmatters[k]) / 2)
        if modi > csize:
            print(k, ' is unknown from ', modi * 5, '%.')
            for j in range(modi - csize):
                for matters in learningmatters[k][lm_half:]:
                    ltar[j].append(k), lcon[j].append(matters)
        else:
            for j in range(size):
                for matters in learningmatters[k][lm_half:]:
                    ltar[j].append(k), lcon[j].append(matters)

        # testing은 모든 class의 data를 다 넣습니다.
        for j in range(size):
            for matters in learningmatters[k][:lm_half]:
                etar[j].append(k), econ[j].append(matters)

    return lcon, ltar, econ, etar


def folding_160617_half(content, target, args=()):
    """
    아직 안 만듬

    - 클래스별로 나눠서 학습시킨다.
    - unknown을 정해서 일부 class를 unknown으로 설정하며 fold한다.
    - sequence for test 길이를 다르게 만든다.
      10, 25, 50개씩 나눠서 학습시킨다.
    :param list content: array-like. it contains content for learning
    :param list target: array-like. it contains target for learning
    :param args:
    :return:
        content for learing,
        target for learning,
        content for examinating,
        target for examinating
    """
    steps = (5, 2, 1)
    size = sum(steps)
    lcon, ltar = [[] for _ in range(size)], [[] for _ in range(size)]
    econ, etar = [[] for _ in range(size)], [[] for _ in range(size)]

    learningmatters = classionary(content, target)

    for i, k in enumerate(learningmatters):
        lm_half = int(len(learningmatters[k]) / 2)
        if len([x for x in args if x in k]) == 0:  # if known
            for matters in learningmatters[k][lm_half:]:
                for x in range(size):
                    ltar[x].append(k), lcon[x].append(matters)
        i = 0
        exammatters = list(learningmatters[k][:lm_half])
        random.shuffle(exammatters)
        for subsize in steps:
            for subexam in chunks(exammatters, subsize):
                for matters in subexam:
                    etar[i].append(k), econ[i].append(matters)
                i += 1

    return lcon, ltar, econ, etar


def folding_161101_half(content, target, args=()):
    """
    지정한 class만 unknown으로 만듭니다.
    그리고 5개의 fold는 test data의 갯수만 달라집니다.
    50, 40, 30, 20, 10개로 나눠집니다.

    :param list content: array-like. it contains content for learning
    :param list target: array-like. it contains target for learning
    :param tuple args: array-like. 특정 클래스 이름
    :return:
        content for learing,
        target for learning,
        content for examinating,
        target for examinating
    """

    return lcon, ltar, econ, etar


def plot_lines(filename, title, *args):
    """
    그래프를 선으로 그립니다.
    :param filename: 파일명
    :param title: 그래프 제목
    :param args: 그래프를 그릴 자료
    이 자료는 dictionary이며, key로 'x', 'y', 'name'을 받습니다.
    :return:엄ㅋ슴ㅋ
    """
    fig, ax = plt.subplots()
    plt.title(title)
    for xy in args:
        x = xy['x']
        y = xy['y']
        plt.plot(x, y, label=xy['name'])
    plt.legend()
    fig.savefig(filename + '.png')
    plt.clf()
    pass


def call_recursive(call: callable, data: list):
    """
    어떤 함수가 있는데, 그 함수가 1-D array만을 처리할 수 있을 때

    multi-D array를 처리할 수 있는 함수입니다.
    또한 내부적으로 row 중심으로 처리하지 않고 column중심으로 처리합니다.

    예를 들어 [1,2,3;4,5,6]이 있고, 이 함수를 사용해 mean을 구한다면
    [2.5, 3.5, 4.5]가 반환됩니다.

    :param callable call: function
    :param list data:
    :return:
    """
    datype = type(data[0])
    if datype is list or datype is tuple:
        res = [call_recursive(call, x) for x in zip(*data)]
        return res
    else:
        return call(data)


def pickme(data: list, size: int):
    """
    data에서 size갯수만큼 임의로 선택하여 list를 리턴합니다.
    :param data:
    :param size:
    :return:
    """
    _data = data
    random.shuffle(_data)
    return [x for x in _data[:size]]


def parachuteme(data: list, size: int):
    """
    data에서 size갯수만큼 앞에서부터 선택하여 list를 리턴합니다.
    :param data:
    :param size:
    :return:
    """
    _data = data
    return [x for x in _data[:size]]


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    :param n:
    :param l:
    """
    step = int(len(l) / n)
    for i in range(n):
        yield l[i * step:(i + 1) * step]
