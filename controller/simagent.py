from model.probaclass_network import CPON
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np
import copy
import random
import controller
import time

default_fold = 2


class LearningManager:
    """
    학습에 필요한 자료를 관리합니다.
    """

    def __init__(self, fold):
        self._ocontent, self._otarget = [], []
        self._lcontent, self._ltarget = [], []
        self._econtent, self._etarget = [], []
        self._fold = fold

    def uploadlearn(self, content, target):
        """
        학습 데이터를 업로드합니다.

        :param content: array-like
        :param target: array-like
        :return: None
        """
        self._lcontent, self._ltarget = content, target

    def uploadexam(self, content, target):
        """
        테스트 데이터를 업로드합니다.

        :param content: array-like
        :param target: array-like
        :return:
        """
        self._econtent, self._etarget = content, target

    def learnsource(self):
        """
        학습 데이터를 각 fold마다 yield합니다.

        :yield:
            learning content: array-like \n
            learning target: array-like
        """
        for lc, lt in zip(self._lcontent, self._ltarget):
            yield lc, lt

    def examinatesource(self):
        """
        테스트 데이터를 각 fold마다 yield합니다.

        :yield:
            examinate content: array-like \n
            examinate target: array-like
       """
        for ec, et in zip(self._econtent, self._etarget):
            yield ec, et

    def fit(self, data, target):
        """
        fold갯수만큼 나눕니다.
        :param data:
        :param target:
        :return:
        """
        ldict = {x: [] for x in set(target)}

        for d, t in zip(data, target):
            ldict[t].append(d)

        for i in range(self._fold):
            lc, lt, ec, et = [], [], [], []
            for key, values in ldict.items():
                subvalues = [x for x in values]
                lentotal = len(subvalues)
                kec = []
                while len(kec) != int(lentotal / self._fold):
                    i = random.randrange(lentotal - len(kec))
                    kec.append(subvalues.pop(i))
                ec.extend(kec), et.extend([key for _ in kec])
                lc.extend(subvalues), lt.extend([key for _ in subvalues])
            self._lcontent.append(lc), self._ltarget.append(lt)
            self._econtent.append(ec), self._etarget.append(et)
        pass

    def source(self):
        """
        학습 및 테스트 데이터를 각 fold마다 yield합니다.

        :yield:
            learning content: array-like \n
            learning target: array-like \n
            examinate content: array-like \n
            examinate target: array-like
        """
        for lc, lt, ec, et in zip(self._lcontent, self._ltarget, self._econtent, self._etarget):
            yield lc, lt, ec, et
    pass


class SimAgent:
    """
    Design Purpose
    --------------
    1. fold learning resource
    2. manage sim
    3. manage learning data and test data
    4. manage test result of simulorules
    """

    def __init__(self, fold=2):
        self.simtaglist = []  # list of classifier
        self._fold = fold  # size of fold
        self._data, self._target = [], []
        self._fold_data, self._fold_target = [], []
        self._learn_data, self._learn_target = [], []
        self._pred_data, self._pred_target = [], []
        self._lm = LearningManager(fold)
        self.unknown = False

    @property
    def folder(self):
        return self._lm

    @folder.setter
    def folder(self, learningmanager):
        self._lm = learningmanager

    def addsim(self, simtag):
        """
        add classifier simulorule with passing by object type check.
        :param simtag:
        :return:
        """
        if isinstance(simtag, Sim):
            self.simtaglist.append(simtag)
        else:
            print('please input SimTag')
        pass

    def fit(self, data, target):
        """
        :param data: {array-like, sparce matrix}, shape = [n_samples, n_features]
            For kernel="precomputed", the expected shape of X is
            [n_samples_test, n_samples_train]

        :param target: array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)
        :return: ClfSim
        """
        lendata, lentarget = len(data), len(target)

        if lendata != lentarget:
            print('Length of X and y are different.')
            return 0
        elif lendata % self._fold != 0:
            print('Length of data is not compitible with fold. modulus is dropped.')

        f = self._fold
        fold_data, fold_target = [[] for _ in range(f)], [[] for _ in range(f)]

        if self.unknown:
            f, self._fold = 6, 6
            # 먼저 반반으로 나눠서 training과 testing이란 이름으로 저장하고
            # 첫번째껀 그냥 저장하고 2번째부터 6번째까지는 5개씩 줄여서 저장
            # 따라서 첫번째는 50개의 class, 6번째는 25개의 class가 저장됨
            # 이젠 여기서 folding을 과정을 모두 처리함
            fold_data, fold_target = [[list(), list()] for _ in range(f)], [[list(), list()] for _ in range(f)]
            targetbuffer = [x for x in set(target)]
            targetbuffer.sort()

            inputdict = {t: [] for t in targetbuffer}
            for d, t in zip(data, target):
                inputdict[t].append(d)

            for i, t in enumerate(targetbuffer):
                lend = len(inputdict[t])
                learndata, testdata = inputdict[t][:int(lend / 2)], inputdict[t][int(lend / 2):]
                # targetindex = int(t)  # 구버전용
                # targetindex = int(t[2:])  # 구버전용(EPxxxx로 naming된 data)
                foldindex_upbound = int(i / 5) + 1
                # index of second-level array: 0 is training, 1 is testing data or target.
                for foldindex in range(6):
                    fold_data[foldindex][1].extend(learndata)
                    fold_target[foldindex][1].extend([t for _ in learndata])
                    if foldindex < foldindex_upbound:
                        fold_data[foldindex][0].extend(learndata)
                        fold_target[foldindex][0].extend([t for _ in learndata])
        else:
            for target_index, zxy in enumerate(list(zip(data, target))):
                foldindex = target_index % f
                fold_data[foldindex].append(zxy[0])
                fold_target[foldindex].append(zxy[1])

        self._fold_data, self._fold_target = fold_data, fold_target
        self._data, self._target = data, target

        return self

    def folding(self):
        fold_data, fold_target, f = self._fold_data, self._fold_target, self._fold

        if self.unknown:
            for j in range(f):
                # index of second-level array: 0 is training, 1 is testing data or target.
                ldata, ltarget = fold_data[j][0], fold_target[j][0]  # learning data
                known_target = set(ltarget)
                pdata, ptarget = fold_data[j][1], [x if x in known_target else 'unknown' for x in fold_target[j][1]]  # predicting data

                self._learn_data, self._learn_target = ldata, ltarget
                self._pred_data, self._pred_target = pdata, ptarget

                yield ldata, ltarget, pdata, ptarget
        else:
            for j in range(f):
                ldata, ltarget = [], []  # learning data
                pdata, ptarget = None, None  # predicting data

                self._learn_data, self._learn_target = ldata, ltarget
                self._pred_data, self._pred_target = pdata, ptarget

                yield ldata, ltarget, pdata, ptarget

    def simulate(self):
        f = 1
        # for fld, flt, fpd, fpt in self.folding():
        for ld, lt, ed, et in self._lm.source():
            for simtag in self.simtaglist:
                simtag.simulate(ld, lt, ed, et)
                # simtag.simulate_160518(ld, lt, ed, et)
            print("fold%02d complete" % f)
            f += 1

        for simtag in self.simtaglist:
            ave_stats(simtag)

        return self.simtaglist

    def simulate_unknown(self):
        f = 1
        # for fld, flt, fpd, fpt in self.folding():
        for ld, lt, ed, et in self._lm.source():
            for simtag in self.simtaglist:
                # simtag.simulate_unknown_160329(ld, lt, ed, et)
                simtag.simulate_unknown_160520(ld, lt, ed, et)
            print("fold%02d complete" % f)
            f += 1

        for simtag in self.simtaglist:
            ave_stats(simtag)

        return self.simtaglist


class Sim:
    """
    각 classifier를 simulate하고, 그 결과를 정리합니다.
    """
    def __init__(self, simulor, simulorname: str):
        self.simulor, self.simulorname = simulor, simulorname
        self.predlist = []
        self.statistics = {}
        self.pval_list = []
        self.testtarget = []
        self._fold = 0

    @staticmethod
    def factory(clfname):
        """
        py:class::`SimAgent`에서 관리할 수 있는 Classifier는 생성해줍니다.

        :param str clfname:
        :return: new SigAgent
        """

        sim = SVC()
        return Sim(sim, clfname)

    def simulate(self, fit_data, fit_target, pred_data, pred_target):
        """
        :param fit_data: data for fit
        :param fit_target: target for fit
        :param pred_data: data for predict
        :param pred_target: target for messurement
        :return:self
        fit, predict and measure statistics on each fold
        """
        self.simulor.fit(fit_data, fit_target)
        pred = self.simulor.predict(pred_data)
        stats = self.statistics
        self.testtarget.append(pred_target)
        # print(len([x for x in pred if 'REJECT' in x]))
        # print(len([x for x, y in zip(pred, pred_target) if 'REJECT' in x and '80' in y]))

        # if type(self.simulor) == CPON:
        #     self.simulor.pred_pval

        self.predlist.append(pred)
        # TODO measurement regist
        update_stats(stats, 'acc', metrics.accuracy_score(pred_target, pred, normalize=True))
        update_stats(stats, 'p_a', metrics.precision_score(pred_target, pred, average='macro'))
        # update_stats(stats, 'p_i', metrics.precision_score(pred_target, pred, average=None))
        update_stats(stats, 'r_a', metrics.recall_score(pred_target, pred, average='macro'))  # NOT COMPITIBLE FOR MULTI-CLASS CLASSIFICATION)
        # update_stats(stats, 'r_i', metrics.recall_score(pred_target, pred, average=None))  # NOT COMPITIBLE FOR MULTI-CLASS CLASSIFICATION)
        update_stats(stats, 'f_a', metrics.f1_score(pred_target, pred, average='macro'))   # NOT COMPITIBLE FOR MULTI-CLASS CLASSIFICATION
        # update_stats(stats, 'f_i', metrics.f1_score(pred, pred_target, average='micro'))   # NOT COMPITIBLE FOR MULTI-CLASS CLASSIFICATION
        # update_stats(stats, 'rec', metrics.recall_score(pred, pred_target, average='binary', pos_label=pred_target[0]))  # binary
        # update_stats(stats, 'pre', metrics.precision_score(pred_target, pred, average='binary', pos_label=pred_target[0]))  # binary
        # update_stats(stats, 'f1m', metrics.f1_score(pred, pred_target, average='binary', pos_label=pred_target[0]))   # bianry
        # update_stats(stats, 'scores', confusion_matrix(pred_target, pred))
        # update_stats(stats, 'dtd', detectability(pred_target, pred))
        # update_stats(stats, 'unp', unknown_precision(pred_target, pred))

        return self

    def simulate_unknown(self, fit_data, fit_target, pred_data, pred_target):
        """
        unknown검출용 simulate
        classinary해서 predict한 후 pred로 모아서 계산한다.
        unknown으로 지정된 class는 prefix로 'u'를 붙인다.
        :param fit_data:
        :param fit_target:
        :param pred_data:
        :param pred_target:
        :return:
        """
        self._fold += 1
        self.simulor.fit(fit_data, fit_target)
        pdict = controller.classionary(pred_data, pred_target)
        stats = self.statistics
        pretag, dtstag, seqtag = [], [], []

        with open('sequence_of_pvalue' + "_{}".format(str(self._fold)) + '.csv', 'w') as f:
            for k in pdict:
                idtt = True if k in fit_target else False
                tag = k if idtt else 'u{}'.format(k)
                dts_pred = self.simulor.predict(pdict[k])

                if idtt:
                    sqp = [x[tag] for x in self.simulor.classoutputs_network]
                else:
                    sqp = [max(x.values()) for x in self.simulor.classoutputs_network]
                m, v = np.mean(sqp), np.var(sqp, ddof=1)
                seq_pred = []

                if controller.detect_160323(m, len(sqp)):
                    seq_pred = ['u{}'.format(k) for _ in dts_pred]
                else:
                    seq_pred = ['known' if 'u' in tag else tag for _ in dts_pred]
                pretag.extend([tag for _ in dts_pred]), dtstag.extend(dts_pred), seqtag.extend(seq_pred)
                f.write(str(tag) + ',' + str(m) + ',' + str(v) + ',' + str(controller.detect_160323(m, len(sqp))) + '\n')

        testtarget, predtarget = pretag, dtstag
        update_stats(stats, 'dsa', metrics.accuracy_score(pretag, dtstag))
        update_stats(stats, 'dsr', unknown_recall(pretag, dtstag))
        update_stats(stats, 'dsp', unknown_precision(pretag, dtstag))
        update_stats(stats, 'sqa', metrics.accuracy_score(pretag, seqtag))
        update_stats(stats, 'sqr', unknown_recall(pretag, seqtag))
        update_stats(stats, 'sqp', unknown_precision(pretag, seqtag))
        print('dts acc:{}'.format(stats['dsa']['fold'][self._fold - 1]))
        print('dts rec:{}'.format(stats['dsr']['fold'][self._fold - 1]))
        print('dts pre:{}'.format(stats['dsp']['fold'][self._fold - 1]))
        print('seq acc:{}'.format(stats['sqa']['fold'][self._fold - 1]))
        print('seq rec:{}'.format(stats['sqr']['fold'][self._fold - 1]))
        print('seq pre:{}'.format(stats['sqp']['fold'][self._fold - 1]))
        pass

    pass

    def simulate_unknown_160329(self, fit_data, fit_target, pred_data, pred_target):
        """
        unknown검출용 simulate
        classinary해서 predict한 후 pred로 모아서 계산한다.
        unknown으로 지정된 class는 prefix로 'u'를 붙인다.

        unknown이 uniform이 아닌 gaussian이라는 가정 하에 ztest를 수행한다.
        :param fit_data:
        :param fit_target:
        :param pred_data:
        :param pred_target:
        :return:
        """
        self._fold += 1
        self.simulor.fit(fit_data, fit_target)
        pdict = controller.classionary(pred_data, pred_target)
        stats = self.statistics
        pretag, dtstag, prstag, seqtag = [], [], [], []
        with open("seq_ztest_snd_{}.csv".format(str(self._fold)), 'w') as f:
            for k in pdict:
                seqmap = {x: [] for x in self.simulor.pcnetwork}

                idtt = True if k in fit_target else False
                tag = k if idtt else 'unknown'
                dts_pred = self.simulor.predict(pdict[k])

                for i, output in enumerate(self.simulor.classoutputs_network):
                    for key in output:
                        seqmap[key].append(output[key])

                seqmvd = {x: {} for x in self.simulor.pcnetwork}
                for key in seqmvd:
                    seqmvd[key] = {
                        'm': np.mean(seqmap[key]),
                        'v': np.var(seqmap[key], ddof=1),
                        # 'd': controller.detect_160329(np.mean(seqmap[key]), beta_mean, beta_std, len(seqmap[key]))
                        'd': controller.detect_160323(np.mean(seqmap[key]), len(seqmap[key]))
                    }

                detectlen = len([seqmvd[x]['d'] for x in seqmvd if seqmvd[x]['d']])
                print("{}:".format(tag),
                      "Ambiguous:{}".format([x for x in seqmvd if seqmvd[x]['d']])
                      if detectlen > 1 else "Unknown" if detectlen < 1 else
                      "Accept:{}".format([x for x in seqmvd if seqmvd[x]['d']]))

                key = controller.mymin(seqmvd, key=lambda x: abs(seqmvd[x]['m'] - .5))
                ciresult = controller.detect_160323(seqmvd[key]['m'], len(seqmap[key]))
                detecttag = key if ciresult else 'unknown'
                detect_m = seqmvd[key]['m']
                detect_v = seqmvd[key]['v']

                # pretag.extend([tag[0] for _ in pdict[k]])
                # dtstag.extend([x[0] for x in dts_pred])
                pretag.extend([tag for _ in pdict[k]])
                dtstag.extend(dts_pred)
                _tag = 'unknown' if 'u' in tag else tag
                prstag.extend([_tag[0] for _ in pdict[k]])
                seqtag.extend([detecttag[0] for _ in pdict[k]])
                f.write(','.join((str(k) if idtt else 'u{}'.format(k), repr(detect_m), repr(detect_v), repr(ciresult) if ciresult else repr(ciresult) + ',' + detecttag)) + '\n')
                # fileout_pvalue = open("pval_{}_{}.csv".format(str(self._fold), k), 'w')
                # pvalues = '\n'.join([str(x[key]) for x in self.simulor.classoutputs_network])
                # fileout_pvalue.write(pvalues)
                # fileout_pvalue.close()

        update_stats(stats, 'dsa', metrics.accuracy_score(pretag, dtstag))
        update_stats(stats, 'dsr', unknown_recall(pretag, dtstag))
        update_stats(stats, 'dsp', unknown_precision(pretag, dtstag))
        update_stats(stats, 'sqa', metrics.accuracy_score(prstag, seqtag))
        update_stats(stats, 'sqr', unknown_recall(prstag, seqtag))
        update_stats(stats, 'sqp', unknown_precision(prstag, seqtag))
        print('dts acc\t{}'.format(stats['dsa']['fold'][self._fold - 1]))
        print('dts rec\t{}'.format(stats['dsr']['fold'][self._fold - 1]))
        print('dts pre\t{}'.format(stats['dsp']['fold'][self._fold - 1]))
        print('seq acc\t{}'.format(stats['sqa']['fold'][self._fold - 1]))
        print('seq rec\t{}'.format(stats['sqr']['fold'][self._fold - 1]))
        print('seq pre\t{}'.format(stats['sqp']['fold'][self._fold - 1]))
        pass

    def simulate_160518(self, fit_data, fit_target, pred_data, pred_target):
        """
        unknown검출용 simulate for general simulator
        classinary해서 predict한 후 pred로 모아서 계산한다.
        unknown으로 지정된 class는 이름울 'unknown'으로 바꾼다.

        :param fit_data:
        :param fit_target:
        :param pred_data:
        :param pred_target:
        :return:
        """
        self._fold += 1
        stats = self.statistics
        self.simulor.fit(fit_data, fit_target)
        rslt_target = self.simulor.predict(pred_data)
        pdict = controller.classionary(fit_data, fit_target)

        update_stats(stats, 'acc', metrics.accuracy_score(pred_target, rslt_target))
        update_stats(stats, 'pre', metrics.precision_score(pred_target, rslt_target, average='macro'))
        update_stats(stats, 'rec', metrics.recall_score(pred_target, rslt_target, average='macro'))
        print('<<{}>>'.format(self.simulorname))
        print('dts acc:{}'.format(stats['acc']['fold'][self._fold - 1]))
        print('dts pre:{}'.format(stats['pre']['fold'][self._fold - 1]))
        print('dts rec:{}'.format(stats['rec']['fold'][self._fold - 1]))
        pass

    def simulate_unknown_160520(self, fit_data, fit_target, pred_data, pred_target):
        """
        unknown검출용 simulate
        classinary해서 predict한 후 pred로 모아서 계산한다.
        unknown으로 지정된 class는 prefix로 'u'를 붙인다.

        unknown이 uniform이 아닌 gaussian이라는 가정 하에 ztest를 수행한다.
        :param fit_data:
        :param fit_target:
        :param pred_data:
        :param pred_target:
        :return:
        """
        self._fold += 1
        self.simulor.fit(fit_data, fit_target)
        pdict = controller.classionary(pred_data, pred_target)
        stats = self.statistics
        pretag, dtstag, prstag, seqtag = [], [], [], []
        with open("sequence_statistics_{}_{}.csv".format(str(self._fold), time.time()), 'w') as f:
        # with open("sequence_statistics_{}.csv".format(str(self._fold), time.time()), 'w') as f:
            for k in pdict:
                idtt = True if k in fit_target else False
                tag_metrics = k if idtt else 'u'
                tag_print = k if idtt else 'u{}'.format(k)
                ks, dts_pred = self.simulor.predict_ks(pdict[k])
                ssspd = {x: [dts_pred.count(x)] for x in set(dts_pred)}
                sssp = self.simulor.sequence_pvalue
                ssspk = max(ssspd, key=lambda x: ssspd[x])

                if 'u' in ssspk:
                    sp, sm = 0., 0.
                    seqtag.extend(['f' for _ in pdict[k]])
                    accept = False
                else:
                    sp = sssp[ssspk]
                    sscn = self.simulor.classoutputs_network

                    sm = sum([x[ssspk] for x in sscn]) / 50
                    accept = controller.detect_160524(sp, sm, 50)
                    seqtag.extend(['t' if accept else 'f' for _ in pdict[k]])

                print(tag_print, '\t', sp, '\t', sm, '\t', ssspk, accept)
                f.write(','.join([tag_print, repr(sp), repr(sm), ssspk]) + '\n')

                prstag.extend(['f' if 'u' in tag_metrics else 't' for _ in pdict[k]])
                dtstag.extend(['f' if 'u' in x else 't' for x in dts_pred])

        update_stats(stats, 'dsa', metrics.accuracy_score(prstag, dtstag))
        update_stats(stats, 'dsr', metrics.precision_score(prstag, dtstag, average='binary', pos_label='t'))
        update_stats(stats, 'dsp', metrics.recall_score(prstag, dtstag, average='binary', pos_label='t'))
        update_stats(stats, 'dsf', metrics.f1_score(prstag, dtstag, average='binary', pos_label='t'))
        update_stats(stats, 'sqa', metrics.accuracy_score(prstag, seqtag))
        update_stats(stats, 'sqr', metrics.precision_score(prstag, seqtag, average='binary', pos_label='t'))
        update_stats(stats, 'sqp', metrics.recall_score(prstag, seqtag, average='binary', pos_label='t'))
        update_stats(stats, 'sqf', metrics.f1_score(prstag, seqtag, average='binary', pos_label='t'))
        print('dts acc\t{}'.format(stats['dsa']['fold'][self._fold - 1]))
        print('dts pre\t{}'.format(stats['dsp']['fold'][self._fold - 1]))
        print('dts rec\t{}'.format(stats['dsr']['fold'][self._fold - 1]))
        print('dts f1m\t{}'.format(stats['dsf']['fold'][self._fold - 1]))
        print('seq acc\t{}'.format(stats['sqa']['fold'][self._fold - 1]))
        print('seq pre\t{}'.format(stats['sqp']['fold'][self._fold - 1]))
        print('seq rec\t{}'.format(stats['sqr']['fold'][self._fold - 1]))
        print('seq f1m\t{}'.format(stats['sqf']['fold'][self._fold - 1]))
        pass
    pass


def clffactory(clfname, **kwargs):
    """
    generate classifier by name.
    Options are setted on general.

    :param clfname: string,  The name of clssifierz
    :return: SimTag object
    """

    clfname = clfname.lower()
    clf = None

    if ('svm' in clfname) or ('svc' in clfname):
        k = kwargs['kernel'] if 'kernel' in kwargs else 'rbf'
        g = kwargs['gamma'] if 'gamma' in kwargs else 50
        clf = SVC(kernel=k, gamma=g)
        print("[clf factory]clf=\'SVC\', kernel=\'" + clf.kernel + "\', gamma=\'" + repr(clf.gamma) + "\'")
    elif 'knn' in clfname:
        w = kwargs['weights'] if 'weights' in kwargs else 'distance'
        clf = KNeighborsClassifier(weights=w)
        print("[clf factory]clf=\'kNN\', weights=\'" + clf.weights + "\'")
    elif 'cpon' in clfname:
        c = kwargs['cluster'] if 'cluster' in kwargs else 'lk'
        s = kwargs['bse'] if 'betashape' in kwargs else 'mm'
        b = kwargs['beta'] if 'beta' in kwargs else 'scipy'
        k = kwargs['kernel'] if 'kernel' in kwargs else 'gaussian'
        clf = CPON()
        print(clf)
    elif 'rf' in clfname:
        n = kwargs['n_estimators'] if 'n_estimators' in kwargs else 10
        r = kwargs['random_state'] if 'random_state' in kwargs else 33
        clf = RandomForestClassifier(n_estimators=n, random_state=r)
        pass
    elif 'nb' in clfname:
         #clf = GaussianNB()
        pass
    mi = Sim(clf, clfname)

    return mi


def ave_stats(simagent: Sim):
    """

    calculate mean of each statistic and append at the end of lsit of statistic.

    :param simagent: object SimTag
    :return: average of statistics
    """
    for key, struct in simagent.statistics.items():
        struct['average'] = np.mean(struct['fold'])

    return simagent


form_stats = {'fold': [], 'average': 0.0}  # data structure for statistic measurement


def update_stats(pedia: dict, key, value):
    """
    :param pedia: statistics dictionary of classification
    :param key: abbrivation of measurement
    :param value: measurement function
    :return wiki: statistics dictionary of classification
    """
    if key not in pedia:
        pedia[key] = copy.deepcopy(form_stats)

    pedia[key]['fold'].append(value)

    return pedia


def confusion_matrix(target, pred):
    """
    confusion matrix
    :param target:
    :param pred:
    :return:
    """
    cm_dict = [ConfusionMatrix(x) for x in set(target)]

    for t, p in zip(target, pred):
        if t == p:
            for cm in cm_dict:
                if cm.name == p:
                    cm['tp'] += 1
                else:
                    cm['tn'] += 1
        else:
            for cm in cm_dict:
                if cm.name == p:
                    cm['fp'] += 1
                else:
                    cm['fn'] += 1
    return cm_dict


class ConfusionMatrix:
    """
    TP, TN, FP, FN을 계산합니다.
    그리고 이를 통해 AFPR을 계산합니다.
    """
    def __init__(self, name):
        self.name = name
        self.matrix = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __getitem__(self, item):
        return self.matrix[item]

    def measure(self):
        return self.accuracy(), self.precision(), self.recall(), self.f1measure(), self.exactmatch()

    def accuracy(self):
        if 'accuracy' in self:
            acc = (self['tp'] + self['fn']) / (self['tp'] + self['tn'] + self['fp'] + self['fn'])
            self['accuracy'] = acc
        return self['accuracy']

    def precision(self):
        if 'precision' in self:
            pre = self['tp'] / (self['tp'] + self['fp'])
            self['precision'] = pre
        return self['precision']

    def recall(self):
        if 'recall' in self:
            rec = self['tp'] / (self['tp'] + self['fn'])
            self['recall'] = rec
        return self['recall']

    def f1measure(self):
        if 'f1measure' in self:
            f1m = (2 * self['tp']) / (2 * self['tp'] + self['fp'] + self['fn'])
            self['f1measure'] = f1m
        return self['f1measure']

    def exactmatch(self):
        pass


def unknown_recall(test_target, pred_target):
    """
    unknown 중에서 올바르게 unknown으로 분류된 비율
    :param test_target:
    :param pred_target:
    :return:
    """
    tp, fn = [], []
    for tt, dt in zip(test_target, pred_target):
        if 'u' in dt and 'u' in tt:
                tp.append([tt, dt])
        elif 'u' not in dt and 'u' in tt:
            fn.append([tt, dt])
    ltp, lfn = len(tp), len(fn)
    unr = ltp / (ltp + lfn) if ltp > 0 or lfn > 0 else 0
    return unr


def unknown_precision(test_target, pred_target):
    """
    unknown으로 분류된 target 중에서 원래 unknown이 분류된 비율
    :param test_target:
    :param pred_target:
    :return:
    """
    tp, fp = [], []
    for tt, dt in zip(test_target, pred_target):
        if 'u' in dt:
            if 'u' in tt:
                tp.append([tt, dt])
            else:
                fp.append([tt, dt])
    ltp, lfp = len(tp), len(fp)

    unp = ltp / (ltp + lfp) if ltp > 0 or lfp > 0 else 0

    return unp
