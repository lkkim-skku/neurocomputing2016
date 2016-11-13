from matplotlib import pyplot
import os
import csv


def rf_csv(dirname):
    def reader(filename):
        nonlocal dirname
        contentee = []
        with open(os.path.join(dirname, filename)+'.csv', 'r') as file:
            readee = csv.reader(file)
            for row in readee:
                contentee.append(row)
        return contentee
    return reader


class Data:
    def __init__(self, xarr):
        self._x = xarr

    def __call__(self):
        return self._x



class PlotParams:
    def __init__(self, rfreadee: list):
        transarr = zip(*rfreadee)
        for arr in transarr:
            setattr(self, arr[0], Data(arr[1:]))


if __name__ == '__main__':
    rfreader = rf_csv('..\\')
    rf_06ft_maxdepth6 = rfreader('rf_06ft_maxdepth6')
    pp_rf_06ft = PlotParams(rf_06ft_maxdepth6)
    legend_rf, = pyplot.plot(pp_rf_06ft.n_estimator(), pp_rf_06ft.rf(), '-', label='Random Forest')
    legend_cp, = pyplot.plot(pp_rf_06ft.n_estimator(), pp_rf_06ft.cpon(), '--', label='CPON')
    pyplot.xlabel('Number of decision trees')
    pyplot.ylabel('Average EMR')
    pyplot.legend(handles=[legend_rf, legend_cp], loc=4)
    # pyplot.show()
    pyplot.savefig('rf_06ft_maxdepth6.png')
    pyplot.clf()

    rf_12ft_maxdepth6 = rfreader('rf_12ft_maxdepth6')
    pp_rf_12ft = PlotParams(rf_12ft_maxdepth6)
    legend_rf, = pyplot.plot(pp_rf_12ft.n_estimator(), pp_rf_12ft.rf(), '-', label='Random Forest')
    legend_cp, = pyplot.plot(pp_rf_12ft.n_estimator(), pp_rf_12ft.cpon(), '--', label='CPON')
    pyplot.xlabel('Number of decision trees')
    pyplot.ylabel('Average EMR')
    pyplot.legend(handles=[legend_rf, legend_cp], loc=4)
    pyplot.savefig('rf_12ft_maxdepth6.png')
    pyplot.clf()
    pass