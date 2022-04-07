import pandas as pd
import numpy as np


def create_df(col_names, data=None, name=None):
    '''pd.DataFrame(
    [('Ziggy Stardust', 1), ('Aladdin Sane', 1), ('Pin Ups', 1)],
    columns=['title','toprank'])'''
    if data is None:
        df = pd.DataFrame(columns=col_names)
    else:
        df = pd.DataFrame({col: data for col in col_names})

    if name is not None:
        df.name = name
    return  df
