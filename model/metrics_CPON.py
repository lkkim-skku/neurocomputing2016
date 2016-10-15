"""
asdadad
"""


def sequence_of_pvalue(target, p):
    """
    target별로 p-value를 묶어서 표시합니다.
    :param target: target
    :param p: p-value
    :return:
    """
    seqs = {}
    for t, v in zip(target, p):
        if t not in seqs:
            seqs[t] = []
        seqs[t].append(v[t])

    return seqs
