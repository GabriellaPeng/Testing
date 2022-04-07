import numpy as np
import matplotlib.pyplot as plt


# calculate mean, variance: df.describe
def find_nans(df):
    '''df[colname].isna() | df[colname].notna()'''
    mask_nums = df.isna().any()  # 每个col的是否有nan
    sum_nan = df.isna().sum()  # count missing values by columns 每个col的nan的数量
    total_nans = df.isna().sum().sum()  # count totall missing values
    return {'nans by column': mask_nums, 'Sum nans by column': sum_nan,
            'total nums': total_nans}


def dropnans(df, drop_colnames=None, inplace=True, reset_index=True):
    '''

    :param df:
    :param drop_colnames: list of colnames | ['col1', 'col2']
    :param reset_index:
    :return:
    '''
    if drop_colnames is None:
        df = df.dropna(inplace=inplace)  # drop rows with missing values
    else:
        df = df.drop(drop_colnames, axis=1)  # #drop by columns
    if reset_index:
        df.reset_index(inplace=inplace, drop=True)
    return df


def aray_fill_nans(array, method='interpolate', plot=False):
    mask_nans = np.isnan(array)
    goodvals = array[~mask_nans]
    x = np.arange(len(array))

    if method == 'interpolate':
        array = np.interp(x, x[~mask_nans], goodvals)

    elif method == 'frequency':
        counts = np.bincount(goodvals)
        array[mask_nans] = np.argmax(counts)

    elif method == 'average':
        array[mask_nans] = np.nanmean(array)

    if plot:
        if len(goodvals) > 500:
            ms1, ms2 = 0.5, 3
        else:
            ms1, ms2 = None, None

        plot_preanalysis(goodvals, ms1=ms1, mark1='--.',
                         data2=array, c2='orange', mark2='s', ms2=ms2)
    return array


def smooth(array, mask=None, window=10, mode='same', plot=False):
    '''
    mode='same | perform partial correlation on boundaries;
    mode= valid | drop boundary values that cannot be fully correlated
    '''
    if mask is None:
        mask = np.ones(
            window) / window  # default smoothed with uniform mask of length 10;

    y = np.correlate(array, mask, mode)

    if plot:
        ms = [1 if len(array) > 500 else 3]

        plot_preanalysis(array, ms1=ms[0], mark1='o',
                         data2=y, c2='orange', label2='correlated data')

    return y


def df_fill_nans(df, col_names, fillwith='mean'):
    if fillwith == 'mean':
        for col in col_names:
            df[col] = df[col].fillna(df[col].mean())

    elif fillwith == 'median':
        ''
    else:
        for col in col_names:
            df[col] = df[col].fillna(fillwith)


def plot_preanalysis(data, ms1=None, c1=None, mark1=None, label1=None,
                     data2=None, ms2=None, c2=None, mark2=None, label2=None):
    if len(data) > 500:
        plt.rcParams["figure.figsize"] = [30, 8]

    plt.plot(data, mark1, ms=ms1, label=label1, color=c1)

    if data2 is not None:
        x2 = np.arange(len(data2))
        plt.plot(x2, data2, mark2, ms=ms2, color=c2, label=label2)
    plt.legend()
