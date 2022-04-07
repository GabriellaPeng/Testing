

def set_multi_index(df, l_ind_names):
    '''l_ind_names: List of index names|ex ['sex','name','year']'''
    return df.set_index(l_ind_names).sort_index()


def get_multi_index(df, level_name=False, level_value=False):
    '''
    allyears_indexed.loc[('F',claires),:].unstack(level=2)
    "pivot" the third level of the multi-index (years) to create a row of columns;
	result is names (rows) x years (columns)
	:returns indexed_df indexed_df.name | indexed_df.values
    '''
    if level_name:
        indexed_df = df.index.get_level_values(level_name)
        # df.loc[('F', claires), :] or df.loc[:, :, 'F']
    else:
        indexed_df = df.index.get_level_values(level_value)
    return indexed_df


def df_insert_row(df, data, ind=None, iloc=False):
    '''append data=[{col1:val1, col2:val2, col3:val3}];
        loc data = ()
        ind=1 or 2 .. |int'''
    if ind:
        if iloc:
            df.iloc[ind] = data
        else:
            df.loc[ind] = data
    else:
        return df.append(data, ignore_index=True)


## other functions can be used
'''
df.pivot(index=None, columns=None, values=None)
df['date] = pd.to_datetime(df[['year','month','day']])
pd.DatetimeIndex.dayofyear()
pd.concat(pd.read_csv(f'names/yob{year}.txt', names=['name','sex','number']).assign(
    year=year) for year in range(1880, 2019))
df.loc['F',2018]
    .sort_values('number', ascending=False)
    .head(10)
    .reset_index()
    .name

Function:
pd.DataFrame.apply(functions, **kwargs) 

def df_insert_col(to_dframe, insert_data, insert_colname, from_colname=None,
                  from_dframe=None,
                  conditions=None):
    #colnames:str|conditions
    if from_dframe:
        to_dframe[insert_colname] = f(from_dframe[vars(from_colname)])

    elif isinstance(insert_data, dict):
        to_dframe = to_dframe.assign(**{insert_data})
    return to_dframe
    
def df_query(df, colname, select_vals, isin_col=False, contain_vals=False):
    #colname|str
    colname = vars(colname)
    if isin_col:
        df[select_vals in df.colname]  # nobelist is a column name
    elif contain_vals:
        df[df.colname.str.contains(contain_vals)]
    else:
        df[df.colname == select_vals] 
    
np.char.find() # is to return the lowest index at which a certain substring
 
'''
