import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

def ret_kofr(bool):
    readcols = ['KOFR', 'Low', '50th percentile', 'High']
    dir = 'AI_finance_team_01/KOFR/'
    kofr2018 = pd.read_excel(dir + "KOFR2018.xls", index_col = 0, header = 4)[readcols]
    kofr2019 = pd.read_excel(dir + "KOFR2019.xls", index_col = 0, header = 4)[readcols]
    kofr2020 = pd.read_excel(dir + "KOFR2020.xls", index_col = 0, header = 4)[readcols]
    kofr2021 = pd.read_excel(dir + "KOFR2021.xls", index_col = 0, header = 4)[readcols]
    kofr2022 = pd.read_excel(dir + "KOFR2022.xls", index_col = 0, header = 4)[readcols]
    kofr2023 = pd.read_excel(dir + "KOFR2023.xls", index_col = 0, header = 4)[readcols]

    kofr = pd.concat([kofr2018, kofr2019, kofr2020, kofr2021, kofr2022, kofr2023])
    kofr.rename(columns = {'50th percentile' : 'Median'}, inplace = True)
    kofr.index.name = 'Date'
    kofr.index = pd.to_datetime(kofr.index)
    kofr.sort_index(inplace = True)
    return kofr

#print(ret_kofr(bool = True))