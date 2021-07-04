import re

from mailbox_feature_extractor import MailBoxFeatures


class TimeZoneFeatures:
    def __init__(self, ):
        self.open_rate_dict = {}
        self.sbj_rate_dict = {}
        self.count_dict = {}
        self.count_dict_opened = {}
        self.extra_features = {}
        self.mobile_p = MailBoxFeatures()

    def fit(self, df):
        self.open_rate_dict = df.groupby('TimeZone').mean().to_dict()['Opened']
        self.sbj_rate_dict = df.groupby('TimeZone')['Subject'].apply(list).apply(self._rate_func)
        self.count_dict = df.groupby(['TimeZone', 'ContactID']).MailID.count().to_dict()
        self.count_dict_opened = df.groupby(['TimeZone', 'ContactID']).Opened.sum().to_dict()

        self.mobile_p.fit(df)
        df['hour'] = df.TimeZone.apply(self.extracting_hour)
        df['open_rate'] = df['MailBoxID'].apply(lambda x: self.mobile_p.open_rate_dict.get(x, 0))
        df['subj_uniquness'] = df['MailBoxID'].apply(lambda x: self.mobile_p.sbj_rate_dict.get(x, 0))
        df['sent_count_before'] = df.apply(lambda row: self.mobile_p.count_dict.get((row.MailBoxID, row.ContactID), 0), axis=1)
        df['open_count_before'] = df.apply(lambda row: self.mobile_p.count_dict_opened.get((row.MailBoxID, row.ContactID), 0),
                                           axis=1)
        df['user_open_rate'] = df.apply(lambda row: self._save_division(row.open_count_before, row.sent_count_before),
                                        axis=1)
        self.extra_features = df.groupby('hour').mean().to_dict()

    def transform(self, df):
        df['open_rate'] = df['TimeZone'].apply(lambda x: self.open_rate_dict.get(x, 0))
        df['subj_uniquness'] = df['TimeZone'].apply(lambda x: self.sbj_rate_dict.get(x, 0))
        df['sent_count_before'] = df.apply(lambda row: self.count_dict.get((row.TimeZone, row.ContactID), 0), axis=1)
        df['open_count_before'] = df.apply(lambda row: self.count_dict_opened.get((row.TimeZone, row.ContactID), 0),
                                           axis=1)
        df['user_open_rate'] = df.apply(lambda row: self._save_division(row.open_count_before, row.sent_count_before), axis=1)

        df['hour'] = df.TimeZone.apply(self.extracting_hour)
        df['open_rate_2'] = df['hour'].apply(lambda x: self.extra_features['Opened'].get(x, 0))
        df['open_rate_3'] = df['hour'].apply(lambda x: self.extra_features['open_rate'].get(x, 0))
        df['subj_uniquness_2'] = df['hour'].apply(lambda x: self.extra_features['subj_uniquness'].get(x, 0))
        df['sent_count_before_2'] = df['hour'].apply(lambda x: self.extra_features['sent_count_before'].get(x, 0))
        df['open_count_before_2'] = df['hour'].apply(lambda x: self.extra_features['open_count_before'].get(x, 0))
        df['user_open_rate_2'] = df.apply(lambda row: self._save_division(row.open_count_before_2, row.sent_count_before_2),
                                        axis=1)

        return df[['open_rate', 'subj_uniquness', 'sent_count_before', 'open_count_before', 'user_open_rate',
                   'hour', 'open_rate_2', 'open_rate_3', 'subj_uniquness_2', 'sent_count_before_2',
                   'open_count_before_2', 'user_open_rate_2']].values

        # return df[['open_rate', 'subj_uniquness', 'sent_count_before', 'open_count_before', 'user_open_rate',
        #            'hour']].values

    @staticmethod
    def _rate_func(l):
        a = len(set(l)) + 1
        b = len(l) + 1
        return a / b

    @staticmethod
    def _save_division(a, b):
        try:
            return a / b
        except:
            return 0

    @staticmethod
    def extracting_hour(x):
        expr = '[-+]\d\d'
        g = re.findall(expr, str(x))
        try:
            return float(g[0])
        except:
            return 0