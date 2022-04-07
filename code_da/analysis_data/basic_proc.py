import scipy.stats as stats

def read_data(df_data, vars=None):
    '''

    :param df_data:
    :param vars | list: See the distribution of  target variable
    :return:
    '''

    if vars is not None:
        dict_v = {v: df_data[v].value_counts() for v in vars}
        return {'dtype': df_data.dtypes, 'distr_vars': dict_v}
    else:
        return {'dtype': df_data.dtypes}

def drop_cat_features():
    pass
def cont_features():
    pass

def ttest(sample1, sample2):
    tstat, pval = stats.ttest_ind(sample1, sample2, equal_var=False)
    print('t-statistic: {:.1f}, p-value: {:.3}'.format(tstat, pval))

def describe_cont_feature(feature, df, var=None):
    print('\n*** Results for {} ***'.format(feature))
    if var is not None:
        print(df.groupby(var)[feature].describe())
    print(ttest(feature))

