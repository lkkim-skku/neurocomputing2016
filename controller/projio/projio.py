__author__ = 'lkkim'

import xlsxwriter as writer
import os
import time
# from simagent import Sim

directory = "D:\\Document\\niplab\\LIG"
version = "\\july"
trails = "\\output\\"
# dataname = "july_original_TOA"  # 진짜 오리지널 데이터
# dataname = "july_originbeam_TOA"  # 모든 1번 beam의 데이터
# dataname = "july_original_TOA_0102"  # 1번과 2번만
# dataname = "july_original_TOA_7980"  # 1번과 2번만
# dataname = "july_origin_10beam_TOA"  # 모든 beam을 다 나눈 데이터
# dataname = "`naeultech-151208`"  # 나을텍에서 제공해준 output data들
dataname = "april50"  # 맨 처음에 받았던 50개 data
timestamp = time.time()


def load_data():
    with open(os.path.join('source/emitter50.csv'), 'r') as merge:
        data, target = [], []
        for line in merge.readlines():
            line = line.split(',')
            tag, datum = line[0], [float(x) for x in line[1:]]
            target.append(tag)
            data.append(datum)

    return data, target


def import_data():
    data, target = [], []
    conn = connector.connect(host='112.170.132.184', port='8806', user='niplab', password='qwqw`12', database='LIG')
    cursor = conn.cursor()
    # query = "SELECT * from july_originbeam_dTOA"
    query = "SELECT * from " + dataname
    cursor.execute(query)
    for feature in cursor:
        data.append(feature[2:])
        target.append(feature[1])
    return data, target


def measurement(clfmsmtlist: list):
    workbook = initbook('measurement' + repr(timestamp))
    clfsmry_arr = []
    for clfmsmt in clfmsmtlist:
        _clfmsmt = [clfmsmt.simulorname, compound_clfsheet(clfmsmt.simulorname, clfmsmt.statistics, workbook)]
        clfsmry_arr.append(_clfmsmt)
    clfsmry_dict = {}
    for clfsmry in clfsmry_arr:
        keys = clfsmry[1].keys()
        for k in keys:
            if k not in clfsmry_dict:
                clfsmry_dict[k] = {clfsmry[0]: clfsmry[1][k]}
            else:
                clfsmry_dict[k][clfsmry[0]] = clfsmry[1][k]

    table, header = [], ['clf']
    for name in clfsmry_arr:
        header.append(name[0])

    keys = clfsmry_dict.keys()

    for key in keys:
        row = [key]
        clfdict = clfsmry_dict[key]
        row.extend([clfdict[x] for x in header[1:]])
        table.append(row)
    table.sort(key=lambda x: x[0])
    table.insert(0, header)
    table = list(zip(*table))
    sheet = workbook.add_worksheet('summary')
    r, c = 0, 0
    for line in table:
        r, c = worksheet_writeline(sheet, r, c, line)

    workbook.close()
    pass


def worksheet_writeline(worksheet, row, col, line):
    """
    write line
    """
    for cell in line:
        worksheet.write(row, col, cell)
        col += 1
    row += 1
    col = 0

    return row, col


def initbook(bookname):
    dirpath = directory + version + trails + dataname + '_' + repr(timestamp)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    wb = writer.Workbook(dirpath + '\\' + bookname + '.xlsx')

    return wb


def attr_listup(msmtdict: dict, attrname: str):
    """
    :param msmtdict:
    :return:
    """
    attr, ave = [attrname], msmtdict['average']
    attr.extend(msmtdict['fold'])
    attr.append(ave)

    return attr, ave


def compound_clfsheet(simulorname: str, statistics: dict, workbook: writer.Workbook):
    sheetname, stats = simulorname, statistics
    keys = stats.keys()
    msmt_table = []
    clf_smry = {}

    for key in keys:
        msmtdict = stats[key]
        msmt_attr, ave = attr_listup(msmtdict, key)
        msmt_table.append(msmt_attr)
        clf_smry[key] = ave
    msmt_table.sort(key=lambda x: x[0])
    foldattr = ['fold']
    for i in range(1, len(msmt_table[0]) - 1):
        foldattr.append('#%02d' % i)
    foldattr.append('ave')
    msmt_table.insert(0, foldattr)
    msmt_table = list(zip(*msmt_table))
    worksheet = workbook.add_worksheet(sheetname)
    r, c = 0, 0
    for line in msmt_table:
        r, c = worksheet_writeline(worksheet, r, c, line)

    return clf_smry


# def p_value(cponpst: Sim):
#     wb = initbook('p-value')
#     header = ['target', 'predict']
#     tgarr = list(set(cponpst.testtarget[0]))
#     tgarr.sort()
#     header.extend(tgarr)
#     for i, pst in enumerate(list(zip(cponpst.testtarget, cponpst.predlist, cponpst.pval_list))):
#         foldtable = []
#         pst = list(zip(*pst))
#         for rowvalues in pst:
#             pvalarr = [[x['target'], x['p-value']] for x in rowvalues[2]['ppdict']]
#             pvalarr.sort(key=lambda x: x[0])
#             pvalarr = [x[1] for x in pvalarr]
#             row = [rowvalues[0], rowvalues[1]]
#             row.extend(pvalarr)
#             foldtable.append(row)
#
#         foldtable = list(zip(*foldtable))
#         temp = []
#         for row, attr in list(zip(foldtable, header)):
#             temprow = [x for x in row]
#             temprow.insert(0, attr)
#             temp.append(temprow)
#         foldtable = temp
#         foldtable = list(zip(*foldtable))
#         # for predict in pval_list:
#         #     predict['ppdict'].sort(key=lambda x: x['target'])
#
#         ws = wb.add_worksheet('fold#%02d' % (i + 1))
#         r, c = 0, 0
#         for line in foldtable:
#             r, c = worksheet_writeline(ws, r, c, line)
#
#     wb.add_worksheet()
#     wb.close()
