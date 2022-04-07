import os
from tkinter import filedialog
import urllib3
import numpy as np
import pandas as pd

def load_csv(csv_path):
    data = pd.read_csv(csv_path)
    return {'head': data.head(), 'shape': data.shape}

def load_csv_to_df(file=None, colnames=None):
    if not file:
        file = filedialog.askopenfilename()

    df = pd.read_csv(f'{file}', names=colnames)
    return df


def load_text(file=None):
    if not file:
        file = filedialog.askopenfilename()

    if os.path.splitext(file)[1] == '.npy':
        txt_data = np.loadtxt(f'{file}')
    elif os.path.splitext(file)[1] == '.txt':
        txt_data =  [line.strip() for line in open(f'{file}', 'r')]

    return txt_data


def load_np(file=None):
    if not file:
        file = filedialog.askopenfilename(defaultextension='.npy')
    data = np.load(f'{file}')
    return data


def save_df_to_csv(df, file=None, compress=False):
    if not file:
        file = filedialog.asksaveasfilename(defaultextension='.csv')

    if compress:
        name = file.split('/')[-1]
        compression_opts = dict(method='zip',
                                archive_name=f'{name}.csv')
        df.to_csv(f'{name}.zip', index=False, compression=compression_opts)

    if os.path.splitext(file)[-1] != '.csv':
        file = file + 'csv'

    df.to_csv(file, index=False, header=True)


def save_np(data, file=None):
    if not file:
        file = filedialog.asksaveasfilename(defaultextension='.csv')
    np.save(f'{file}.npy', data)


def save_text(file, data, ifnp=True):
    if not file:
        file = filedialog.asksaveasfilename(defaultextension='.txt')

    if ifnp:
        np.savetxt(file, data)


def load_url_data(site_link, l_col_names=None, l_dtype=None):
    #TODO not function yet
    http = urllib3.PoolManager()
    data = http.request('GET', site_link)
    extension = os.path.splitext(data)[1]
    if extension == '.txt':
        data = np.genfromtxt(data, delimiter=None,
                             names=l_col_names,
                             dtype=l_dtype,
                             autostrip=True)
    elif extension == '.csv':
        data = load_csv_to_df(data)
    return data
