import controller
from controller import projio
from controller import simagent
import numpy as np
# from model import metrics_CPON as cmetrics


class SimulationCaseController:
    def __init__(self, fold: int):
        self._fold = fold
        self._agentlist = []
        pass

    @property
    def fold(self):
        return self._fold

    def __getitem__(self, item):
        return self._agentlist[item]

    def fit(self, agent: simagent.SimAgent, **kwargs):
        """
        agent을 등록하고, 두가지 case로 입력된 데이터를 받는다.
        case #1: 그냥 data와 target이 입력된 경우
        그냥 data와 target을 입력한다.
        case #2: lc, lt, ec, et가 지정된 경우
        지정해서 입력한다.
        :param agent:
        :param kwargs:
        :return:
        """
        if 'data' in kwargs and 'target' in kwargs:
            """
            case #1: 그냥 data와 target이 입력된 경우
            그냥 data와 target을 입력한다.
           """
            agent.folder.fit(kwargs['data'], kwargs['target'])
        elif 'lc' in kwargs and 'lt' in kwargs and 'ec' in kwargs and 'et' in kwargs:
            """
            case #2: lc, lt, ec, et가 지정된 경우
            지정해서 입력한다.
            """
            lm = agent.folder
            lm.uploadlearn(kwargs['lc'], kwargs['lt'])
            lm.uploadexam(kwargs['ec'], kwargs['et'])
            agent.folder = lm
        else:
            print('뭔 개짓거리?')
            return False
        self._agentlist.append({'agent': agent})
        pass

    def simulate(self, simulatecase: str):
        for agentdict in self._agentlist:
            sa = agentdict['agent']

            if not isinstance(sa, simagent.SimAgent):
                return False

            if 'known' in simulatecase:
                agentdict['result'] = sa.simulate()

            elif 'identify' in simulatecase:
                agentdict['result'] = sa.simulate_unknown()

            yield agentdict['result']

    @staticmethod
    def factory():
        sa = simagent.SimAgent()
        return sa


if __name__ == "__main__":
    # data, target = projio.load('D:/workshop/NIPLAB/leesuk.kim/raw/50[FREQ,PW,dTOA]') # 실제 논문에서 사용한 dataset (tb.1)
    # data, target = projio.load('D:/workshop/NIPLAB/leesuk.kim/raw/50[FREQ,PW,TOA]') # 실제 논문에서 사용한 dataset (tb.2, tb.3)
    data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/50[FREQ,PW,dTOA]') # 실제 논문에서 사용한 dataset (tb.2, tb.3)
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/50[FREQ,PW,TOA]') # 실제 논문에서 사용한 dataset (tb.1)

    sac = SimulationCaseController(fold=10)

    case30per = (	    ###15 개###
        (	'46',	'27',	'03',	'13',	'20',	'24',	'42',	'07',	'26',	'11',	'40',	'04',	'17',	'45',	'06'	),
        (	'17',	'06',	'40',	'16',	'49',	'04',	'13',	'38',	'07',	'19',	'11',	'30',	'09',	'47',	'08'	),
        (	'10',	'12',	'49',	'02',	'07',	'11',	'37',	'46',	'04',	'01',	'18',	'27',	'48',	'34',	'44'	),
        (	'16',	'09',	'30',	'22',	'21',	'17',	'02',	'03',	'41',	'28',	'14',	'47',	'10',	'07',	'45'	),
        (	'49',	'46',	'43',	'41',	'38',	'25',	'21',	'20',	'14',	'12',	'10',	'08',	'07',	'05',	'02'	),
        (	'50',	'44',	'39',	'37',	'34',	'31',	'27',	'25',	'24',	'23',	'18',	'17',	'10',	'06',	'03'	),
        (	'49',	'42',	'41',	'40',	'37',	'36',	'35',	'34',	'31',	'26',	'22',	'17',	'12',	'11',	'01'	),
        (	'39',	'38',	'37',	'36',	'35',	'33',	'32',	'28',	'26',	'15',	'11',	'10',	'08',	'07',	'02'	),
        (	'46',	'44',	'30',	'25',	'23',	'22',	'21',	'18',	'17',	'16',	'15',	'07',	'05',	'04',	'01'	),
        (	'48',	'46',	'43',	'39',	'33',	'28',	'23',	'18',	'17',	'14',	'07',	'05',	'03',	'02',	'01'	)
    )

    for case in case30per:
        sa = simagent.SimAgent(sac.fold)
        sa.addsim(simagent.clffactory('cpon'))
        lc, lt, ec, et = controller.folding_160411_half(data, target, case)
        sac.fit(sa, lc=lc, lt=lt, ec=ec, et=et)

    sacgen = sac.simulate('identify')
    stats = [x[0].statistics for x in sacgen]
    # for res in sacgen:
    #     agent = res
    #     stats.append(res[0].statistics)  # 0번째 sim의 statistics

    scfdict = {'dsa': [], 'dsp': [], 'dsr': [], 'dsf': [], 'sqa': [], 'sqp': [], 'sqr': [], 'sqf': []}
    for fold in stats:
        for key in fold:
            scfdict[key].append([x for x in fold[key]['fold']])

    for key in scfdict:
        fclist = scfdict[key]
        cflist = [np.average(x) for x in zip(*fclist)]
        scfdict[key] = cflist

    keyset = [k for k in scfdict]
    keyset.sort()
    print('15개\t', '\t'.join(keyset))
    for i in range(5):
        print((i + 1) * 10, '\t', '\t'.join([str(scfdict[k][i]) for k in keyset]))

# agent = simagent.SimAgent(fold=10)  # default : 2
    # cpon = simagent.clffactory('cpon')
    # agent.addsim(cpon)
    # lm = agent.folder
    # agent.folder.fit(data, target)
    # lationresult = agent.simulate()
