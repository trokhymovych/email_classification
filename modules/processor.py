import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing


class Preprocessor:
    def __init__(self, ):
        encoders = {}

    def extracting_time(self, x):
        expr = '(\d+):(\d+)\.(\d+)'
        if x is None:
            return 0, 0, 0
        r = re.search(expr, str(x))
        try:
            return r.group(1), r.group(2), r.group(3)
        except:
            return 0, 0, 0

    def fit_texts(self, X_train):
        texts = X_train['Subject'].fillna("").astype(str)
        self.vectorizer = TfidfVectorizer(max_features=500)
        return self.vectorizer.fit_transform(texts).todense()

    def transform_texts(self, X_test):
        texts = X_test['Subject'].fillna("").astype(str)
        return self.vectorizer.fit_transform(texts).todense()

    def fit_mail_box(self, X_train):
        self.mail_box_encoder = preprocessing.LabelEncoder()
        return self.mail_box_encoder.fit_transform(X_train['MailBoxID']).reshape(-1, 1)

    def transform_mail_box(self, X_train):
        return self.mail_box_encoder.transform(X_train['MailBoxID']).reshape(-1, 1)

    def fit_timezone(self, X_train):
        self.timezone_encoder = preprocessing.LabelEncoder()
        return self.timezone_encoder.fit_transform(X_train['TimeZone']).reshape(-1, 1)

    def transform_timezone(self, X_train):
        return self.timezone_encoder.transform(X_train['TimeZone']).reshape(-1, 1)

    def process_send_on(self, X_train):
        a, b, c = [], [], []
        for s in X_train.SentOn:
            n, nn, nnn = self.extracting_time(s)
            a.append(int(n))
            b.append(int(nn))
            c.append(int(nnn))

        return np.array([a, b, c]).T

    def fit(self, X_train):
        a = self.fit_texts(X_train)
        b = self.fit_mail_box(X_train)
        c = self.fit_timezone(X_train)
        d = self.process_send_on(X_train)

        return np.array(np.hstack([a, b, c, d]), dtype=float)

    def transform(self, X_test):
        a = self.transform_texts(X_test)
        b = self.transform_mail_box(X_test)
        c = self.transform_timezone(X_test)
        d = self.process_send_on(X_test)

        return np.array(np.hstack([a, b, c, d]), dtype=float)
