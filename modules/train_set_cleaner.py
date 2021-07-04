from copy import deepcopy
from datetime import datetime

import pandas as pd


def parse_time_series(row):
    return datetime.strptime(row, '%m/%d/%y %H:%M')


def clean_train_set(train_df: pd.DataFrame, test_df: pd.DataFrame):
    tr = deepcopy(train_df)
    te = deepcopy(test_df)

    te['Opened'] = 2

    mrg = tr.append(te)
    mrg['dt'] = mrg['SentOn']
    mrg['dt'] = mrg['dt'].apply(parse_time_series)

    gb = mrg.groupby('ContactID').sum().sort_values('Opened')
    group1 = [] #list(gb[gb.Opened == 0].index)

    group2 = list(set(tr.ContactID) - set(te.ContactID))

    mail_ids_to_exclude = list(set(group1 + group2))
    tr = tr[~tr.ContactID.isin(mail_ids_to_exclude)]
    tr = tr.reset_index()
    return tr
