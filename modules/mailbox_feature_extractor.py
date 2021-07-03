class MailBoxFeatures:
    def __init__(self, ):
        self.open_rate_dict = {}
        self.sbj_rate_dict = {}
        self.count_dict = {}
        self.count_dict_opened = {}

    def fit(self, df):
        self.open_rate_dict = df.groupby('MailBoxID').mean().to_dict()['Opened']
        self.sbj_rate_dict = df.groupby('MailBoxID')['Subject'].apply(list).apply(self._rate_func)
        self.count_dict = df.groupby(['MailBoxID', 'ContactID']).MailID.count().to_dict()
        self.count_dict_opened = df.groupby(['MailBoxID', 'ContactID']).Opened.sum().to_dict()

    def transform(self, df):
        df['open_rate'] = df['MailBoxID'].apply(lambda x: self.open_rate_dict.get(x, 0))
        df['subj_uniquness'] = df['MailBoxID'].apply(lambda x: self.sbj_rate_dict.get(x, 0))
        df['sent_count_before'] = df.apply(lambda row: self.count_dict.get((row.MailBoxID, row.ContactID), 0), axis=1)
        df['open_count_before'] = df.apply(lambda row: self.count_dict_opened.get((row.MailBoxID, row.ContactID), 0),
                                           axis=1)
        df['user_open_rate'] = df.apply(lambda row: self._save_division(row.open_count_before, row.sent_count_before), axis=1)

        return df[['open_rate', 'subj_uniquness', 'sent_count_before', 'open_count_before', 'user_open_rate']].values

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