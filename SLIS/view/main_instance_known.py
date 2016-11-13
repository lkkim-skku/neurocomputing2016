__author__ = 'lkkim'


import controller
from controller import projio
from controller import simagent
from model import metrics_CPON as cmetrics


if __name__ == '__main__':
    # agent = simagent.SimAgent()  # default : 2
    agent = simagent.SimAgent(fold=10)  # default : 2
    cpon = simagent.clffactory('cpon')
    agent.addsim(cpon)
    # agent.addsim(simagent.clffactory('svm', gamma = 0.03125, decision_function_shape='ovr'))
    # agent.addsim(simagent.clffactory('svm', gamma = 0.0625))
    # agent.addsim(simagent.clffactory('svm', gamma = 0.125))
    # agent.addsim(simagent.clffactory('svm', gamma = 0.25))
    # agent.addsim(simagent.clffactory('svm', gamma = 0.5))
    # agent.addsim(simagent.clffactory('svm', gamma = 1))
    # agent.addsim(simagent.clffactory('svm', gamma = 4))
    agent.addsim(simagent.clffactory('svm'))
    agent.addsim(simagent.clffactory('knn'))
    # agent.addsim(simagent.clffactory('rf'))
    # agent.addsim(simagent.clffactory('nb'))
    # agent.addsim(simagent.clffactory('knn'))
    # agent.addsim(simagent.clffactory('lsvm'))

    # agent.addsim(simagent.clffactory('svm', gamma = 4))
    # agent.addsim(simagent.clffactory('knn', weights='uniform', algorithm='brute'))

    # for i in range(1, 31):
    #     agent.addsim(simagent.clffactory('rf', n_estimators=i, max_features=6, max_depth=6)) # 요거중요
    # for i in range(1, 31):
    #     agent.addsim(simagent.clffactory('rf', n_estimators=i, max_features=12, max_depth=6))  # 요거중요

    # agent.addsim(simagent.clffactory('rf'), max_features=12)
    # agent.addsim(simagent.clffactory('nb')) # 폐기

    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/prev50[FREQ,PW,dTOA]')
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/prev50[FREQ,PW,TOA]')
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/50_6feature[FREQ,PW,dTOA]')
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/50_6feature[FREQ,PW,TOA]')
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/50[FREQ,PW,dTOA]') # 실제 논문에서 사용한 dataset (tb.2, tb.3)
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/50[FREQ,PW,TOA]') # 실제 논문에서 사용한 dataset (tb.1)
    # data, target = projio.load('D:/workshop/NIPLAB/leesuk.kim/raw/50_6feature[FREQ,PW,dTOA]')
    # data, target = projio.load('D:/workshop/NIPLAB/leesuk.kim/raw/50_6feature[FREQ,PW,TOA]')
    data, target = projio.load('D:/workshop/NIPLAB/leesuk.kim/raw/50[FREQ,PW,dTOA]') # 실제 논문에서 사용한 dataset (tb.1)
    # data, target = projio.load('D:/workshop/NIPLAB/leesuk.kim/raw/50[FREQ,PW,TOA]') # 실제 논문에서 사용한 dataset (tb.2, tb.3)

    # 여기서 learning data와 examining data를 관리해줘야 합니다.
    agent.folder.fit(data, target)

    lationresult = agent.simulate()
    # lationresult = agent.simulate_unknown()
    # res = lationresult[0].statistics
    # reskey = list(res.keys())
    # reskey.sort()
    # for key in reskey:
    #     print(key, end='\t')
    #     print('\t'.join([repr(x) for x in res[key]['fold']]))
    projio.measurement(lationresult)

    print("End py")
