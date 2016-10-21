__author__ = 'lkkim'


import controller
from controller import projio
from controller import simagent
from model import metrics_CPON as cmetrics


if __name__ == '__main__':
    agent = simagent.SimAgent()  # default : 2
    # agent = simagent.SimAgent(fold=10)  # default : 2
    cpon = simagent.clffactory('cpon')
    agent.addsim(cpon)
    # agent.addsim(simagent.clffactory('svm', gamma = 0.03125, decision_function_shape='ovr'))
    # agent.addsim(simagent.clffactory('svm', gamma = 0.0625))
    # agent.addsim(simagent.clffactory('svm', gamma = 0.125))
    # agent.addsim(simagent.clffactory('svm', gamma = 0.25))
    # agent.addsim(simagent.clffactory('svm', gamma = 0.5))
    # agent.addsim(simagent.clffactory('svm', gamma = 1))
    # agent.addsim(simagent.clffactory('svm', gamma = 2))
    # agent.addsim(simagent.clffactory('svm', gamma = 4))
    # agent.addsim(simagent.clffactory('svm', gamma = 8))
    # agent.addsim(simagent.clffactory('svm', gamma = 16))
    # agent.addsim(simagent.clffactory('svm', gamma = 32))
    # agent.addsim(simagent.clffactory('svm', gamma = 64))
    # agent.addsim(simagent.clffactory('knn', weights='uniform', algorithm='brute'))
    agent.addsim(simagent.clffactory('svm'))
    agent.addsim(simagent.clffactory('knn'))
    # agent.addsim(simagent.clffactory('rf'))
    # agent.addsim(simagent.clffactory('nb'))

    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/100_B0001[FREQ,PW,dTOA]')
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/prev50[FREQ,PW,dTOA]')
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/prev50[FREQ,PW,TOA]')
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/50[FREQ,PW,dTOA]')
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/50_6feature[FREQ,PW,dTOA]')  # 쓰면 좋을 것 같은 dataset
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/50_6feature[FREQ,PW,TOA]')
    # data, target = projio.load('D:\/orkshop/NIPLAB/raw/50[FREQ,PW,TOA]')
    data, target = projio.load('D:/Workshop/NIPLAB/raw/50[FREQ,PW,dTOA]')
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/50[FREQ,PW,TOA]')  # 실제 논문에서 사용한 dataset
    # data, target = projio.load('D:/Workshop/NIP lab/niplab/14.11-16.04_RadarSignal/Data/input/raw/50[FREQ,PW,TOA,dTOA]')

    # 여기서 learning data와 examining data를 관리해줘야 합니다.
    lm = agent.folder
    agent.folder.fit(data, target)
    # lc, lt, ec, et = controller.folding_160311_half(data, target)  # 0~50% unknown
    # lc, lt, ec, et = controller.folding_160624_half(data, target)  # 0~50% unknown
    #lc, lt, ec, et = controller.folding_160411_half(data, target)  # 지정한 unknown, test 개수 50~10

    # lc, lt, ec, et = controller.folding_160411_half(data, target, ('16', '07', '08', '15', '17', '18', '20', '25', '26', '50'))
    # lc, lt, ec, et = controller.folding_160411_half(data, target, ('16', '07', '08', '15', '17', '18', '20', '25', '26', '50'))
    # lc, lt, ec, et = controller.folding_160617_half(data, target, ('20', '39', '49', '48', '21'))  # 10%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, ('46', '27', '03', '20', '24', '26', '11', '40', '17', '06'))  # 20%
    # lc, lt, ec, et = controller.folding_160617_half(data, target, ('35', '17', '13', '37', '11', '30', '25', '07', '20', '38'))  # 20%

    ###CASE 01###
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'46', '27', '20', '11', '17'	))  # 10%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'46', '27', '03', '20', '24', '26', '11', '40', '17', '06'	))  # 20%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'46', '27', '03', '13', '20', '24', '42', '07', '26', '11', '40', '04', '17', '45', '06'	))  # 30%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'46', '27', '03', '13', '05', '10', '20', '24', '41', '07', '26', '11', '40', '14', '04', '01', '17', '45', '32', '06'	))  # 40%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'46', '27', '03', '13', '05', '37', '10', '20', '31', '24', '42', '07', '25', '22', '26', '11', '40', '14', '04', '01', '35', '17', '45', '32', '06'	))  # 50%
    ###CASE 02###
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'17', '6', '13', '38', '09'	))  # 10%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'17', '06', '16', '49', '13', '38', '07', '11', '09', '08'	))  # 20%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'17', '06', '40', '16', '49', '04', '13', '38', '07', '19', '11', '30', '09', '47', '08'	))  # 30%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'17', '06', '40', '46', '16', '49', '04', '18', '13', '38', '07', '19', '23', '11', '30', '12', '09', '31', '47', '08'	))  # 40%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'17', '06', '40', '46', '03', '34', '16', '49', '04', '18', '13', '38', '07', '19', '23', '26', '25', '11', '30', '12', '09', '31', '47', '08', '48'	))  # 50%
    ###CASE 03###
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'10',	'46',	'02',	'37',	'01'																					))  # 10%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'10',	'12',	'49',	'02',	'07',	'37',	'46',	'01',	'18',	'34'																))  # 20%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'10',	'12',	'49',	'02',	'07',	'11',	'37',	'46',	'04',	'01',	'18',	'27',	'48',	'34',	'44'											))  # 30%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'10',	'12',	'14',	'49',	'02',	'07',	'11',	'08',	'37',	'46',	'06',	'29',	'04',	'01',	'18',	'27',	'48',	'34',	'44',	'32'						))  # 40%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'10',	'12',	'14',	'49',	'02',	'07',	'11',	'08',	'41',	'37',	'46',	'06',	'29',	'04',	'17',	'01',	'18',	'27',	'33',	'48',	'24',	'34',	'44',	'32',	'38'	))  # 50%
    ###CASE 04###
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'09',	'30',	'17',	'49',	'07'																					))  # 10%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'07',	'37',	'08',	'18',	'26',	'02',	'39',	'36',	'11',	'28'																))  # 20%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'16',	'09',	'30',	'22',	'21',	'17',	'02',	'03',	'41',	'28',	'14',	'47',	'10',	'07',	'45'											))  # 30%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'16',	'09',	'30',	'22',	'21',	'25',	'17',	'02',	'06',	'27',	'43',	'41',	'28',	'31',	'14',	'47',	'10',	'32',	'07',	'45'						))  # 40%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'16',	'09',	'30',	'22',	'21',	'25',	'05',	'17',	'02',	'06',	'27',	'43',	'41',	'28',	'31',	'14',	'47',	'10',	'32',	'07',	'45',	'04',	'40',	'38',	'36'	))  # 50%
    ###CASE 05###
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'46',	'21',	'20',	'14',	'02'																					))  # 10%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'49',	'46',	'43',	'21',	'20',	'14',	'10',	'08',	'05',	'02'																))  # 20%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'49',	'46',	'43',	'41',	'38',	'25',	'21',	'20',	'14',	'12',	'10',	'08',	'07',	'05',	'02'											))  # 30%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'49',	'46',	'43',	'41',	'38',	'36',	'32',	'28',	'26',	'25',	'21',	'20',	'19',	'14',	'12',	'10',	'08',	'07',	'05',	'02'						))  # 40%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'49',	'48',	'46',	'43',	'41',	'38',	'36',	'33',	'32',	'28',	'26',	'25',	'21',	'20',	'19',	'16',	'14',	'12',	'11',	'10',	'08',	'07',	'05',	'02',	'01'	))  # 50%
    ###CASE 06###
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'44',	'25',	'22',	'18',	'16'																					))  # 10%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'44',	'25',	'23',	'22',	'18',	'17',	'16',	'15',	'07',	'01'																))  # 20%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'46',	'44',	'30',	'25',	'23',	'22',	'21',	'18',	'17',	'16',	'15',	'07',	'05',	'04',	'01'											))  # 30%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'46',	'45',	'44',	'42',	'37',	'30',	'26',	'25',	'23',	'22',	'21',	'19',	'18',	'17',	'16',	'15',	'07',	'05',	'04',	'01'						))  # 40%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'48',	'46',	'45',	'44',	'42',	'37',	'35',	'34',	'30',	'26',	'25',	'23',	'22',	'21',	'19',	'18',	'17',	'16',	'15',	'13',	'10',	'07',	'05',	'04',	'01'	))  # 50%
    ###CASE 07###
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'31',	'27',	'24',	'17',	'10'																					))  # 10%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'44',	'31',	'27',	'25',	'24',	'23',	'17',	'10',	'06',	'03'																))  # 20%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'50',	'44',	'39',	'37',	'34',	'31',	'27',	'25',	'24',	'23',	'18',	'17',	'10',	'06',	'03'											))  # 30%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'48',	'46',	'43',	'41',	'39',	'33',	'31',	'29',	'28',	'25',	'23',	'18',	'17',	'14',	'13',	'07',	'05',	'03',	'02',	'01'						))  # 40%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'50',	'48',	'47',	'44',	'42',	'39',	'37',	'36',	'34',	'33',	'31',	'30',	'27',	'26',	'25',	'24',	'23',	'21',	'18',	'17',	'10',	'07',	'06',	'05',	'03'	))  # 50%
    ###CASE 08###
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'46',	'33',	'23',	'14',	'03'																					))  # 10%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'48',	'46',	'39',	'33',	'23',	'17',	'14',	'07',	'03',	'02'																))  # 20%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'48',	'46',	'43',	'39',	'33',	'28',	'23',	'18',	'17',	'14',	'07',	'05',	'03',	'02',	'01'											))  # 30%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'50',	'48',	'47',	'44',	'39',	'37',	'36',	'34',	'33',	'31',	'27',	'25',	'24',	'23',	'18',	'17',	'10',	'07',	'06',	'03'						))  # 40%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'48',	'46',	'43',	'41',	'39',	'38',	'36',	'33',	'31',	'29',	'28',	'26',	'25',	'23',	'22',	'18',	'17',	'14',	'13',	'7',	'05',	'04',	'03',	'02',	'01'	))  # 50%
    ###CASE 09###
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'39',	'26',	'11',	'08',	'07'																					))  # 10%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'47',	'45',	'30',	'28',	'22',	'17',	'16',	'09',	'07',	'02'																))  # 20%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'39',	'38',	'37',	'36',	'35',	'33',	'32',	'28',	'26',	'15',	'11',	'10',	'08',	'07',	'02'											))  # 30%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'50',	'42',	'39',	'38',	'37',	'36',	'35',	'33',	'32',	'28',	'26',	'20',	'19',	'15',	'11',	'10',	'08',	'07',	'02',	'01'						))  # 40%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'50',	'43',	'42',	'39',	'38',	'37',	'36',	'35',	'33',	'32',	'28',	'26',	'22',	'20',	'19',	'17',	'15',	'14',	'11',	'10',	'08',	'07',	'05',	'02',	'01'	))  # 50%
    ###CASE 10###
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'37',	'34',	'22',	'11',	'01'																					))  # 10%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'42',	'40',	'37',	'36',	'35',	'34',	'26',	'22',	'11',	'01'																))  # 20%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'49',	'42',	'41',	'40',	'37',	'36',	'35',	'34',	'31',	'26',	'22',	'17',	'12',	'11',	'01'											))  # 30%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'49',	'42',	'41',	'40',	'37',	'36',	'35',	'34',	'33',	'31',	'30',	'26',	'22',	'19',	'18',	'17',	'12',	'11',	'04',	'01'						))  # 40%
    # lc, lt, ec, et = controller.folding_160411_half(data, target, (	'50',	'49',	'42',	'41',	'40',	'38',	'37',	'36',	'35',	'34',	'33',	'31',	'30',	'27',	'26',	'22',	'21',	'20',	'19',	'18',	'17',	'12',	'11',	'04',	'01'	))  # 50%

    # lm.uploadlearn(lc, lt)
    # lm.uploadexam(ec, et)
    # d, t = projio.load('D:/Document/niplab/14.11-16.04_RadarSignal/Data/input/raw/50[FREQ,PW,dTOA]learn')
    # lm.uploadlearn([d], [t])
    # d, t = projio.load('D:/Document/niplab/14.11-16.04_RadarSignal/Data/input/raw/50[FREQ,PW,dTOA]test')
    # lm.uploadexam([d], [t])

    lationresult = agent.simulate()
    # lationresult = agent.simulate_unknown()
    # res = lationresult[0].statistics
    # reskey = list(res.keys())
    # reskey.sort()
    # for key in reskey:
    #     print(key, end='\t')
    #     print('\t'.join([repr(x) for x in res[key]['fold']]))
    projio.measurement(lationresult)
    # for cponpst in lationresult:
    #     if 'cpon' in cponpst.simulorname:
            # cmetrics.sequence_of_pvalue(cponpst.simulor.)
            # projio.p_value(cponpst)
            # print(cponpst.simulor.pred_pval)
            # for i, ppv_fold in enumerate(cponpst.simulor.pred_pval):
            #     print(("fold %02d" % i) + "p-value result")
            #     ppvstr = ""
            #     for ppv_cls in ppv_fold:
            #         ppvstr += ppv_cls + "\t" + repr(ppv_fold[ppv_cls]) + "\t"
            #     print(ppvstr)

    print("End py")
