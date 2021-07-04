from datetime import datetime, timedelta


class ClientFeatures:
    def __init__(self, ):
        self.open_rate_dict = {}
        self.sbj_rate_dict = {}
        self.count_dict = {}
        self.count_dict_opened = {}

    def fit(self, df):
        df['hour'] = df.SentOn.apply(self.extract_hour_strict)
        self.open_rate_dict = df.groupby('ContactID').mean().to_dict()['Opened']
        self.sbj_rate_dict = df.groupby('ContactID')['Subject'].apply(list).apply(self._rate_func)
        self.count_dict = df.groupby(['hour', 'ContactID']).MailID.count().to_dict()
        self.count_dict_opened = df.groupby(['hour', 'ContactID']).Opened.sum().to_dict()

    def transform(self, df):
        df['hour'] = df.SentOn.apply(self.extract_hour_strict)
        df['open_rate'] = df['ContactID'].apply(lambda x: self.open_rate_dict.get(x, 0))
        df['subj_uniquness'] = df['ContactID'].apply(lambda x: self.sbj_rate_dict.get(x, 0))
        df['sent_count_before'] = df.apply(lambda row: self.count_dict.get((row.hour, row.ContactID), 0), axis=1)
        df['open_count_before'] = df.apply(lambda row: self.count_dict_opened.get((row.hour, row.ContactID), 0),
                                           axis=1)
        df['user_open_rate'] = df.apply(lambda row: self._save_division(row.open_count_before, row.sent_count_before), axis=1)

        return df[['open_rate']].values

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
    def extract_hour_strict(time_str: str):
        if time_str in (None, 'none'):
            return 0
        dt = datetime.strptime(time_str, '%m/%d/%y %H:%M')
        return dt.hour

    @staticmethod
    def extract_hour(time_str: str):
        if time_str in (None, 'none'):
            return 0
        dt = datetime.strptime(time_str, '%m/%d/%y %H:%M')
        return dt.hour + dt.minute/60
