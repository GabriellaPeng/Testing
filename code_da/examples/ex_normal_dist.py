import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from code_da.prep_data.prepoc_data import smooth
from data import getweather

allyears = np.arange(1880, 2020)


def plotanomaly(station):
    # grab the data
    alldata = np.vstack([getweather.getyear(station, ['TMIN', 'TMAX'], year)
                         for year in allyears])

    # make yearly averages, and then the midcentury average
    sum_val = alldata[:, :]['TMIN'] + alldata[:, :]['TMAX']

    c = pd.DataFrame(sum_val.T).apltly(pd.Series.last_valid_index)
    fst_valid_year = c[~c.notnull()].index[-1]
    fst_valid_year = allyears[fst_valid_year]
    print(f'First valid year of staion {station} is {allyears[fst_valid_year]}')

    allavg = np.nanmean(0.5 * sum_val[fst_valid_year], axis=1)

    ceil_val = int(np.ceil(allyears.mean() - 5))
    ceil_index = list(allyears).index(ceil_val)
    midcentury = np.nanmean(allavg[ceil_index:ceil_index + 10])

    # plot with smoothing, adding a label that we can show in a legend
    plt.plot(allyears[4:-4], smooth(allavg - midcentury, 9, 'valid'), label=station)

    # set a reasonable range
    plt.axis(ymin=-3, ymax=3)

plotanomaly('NEW YORK')