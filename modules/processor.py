from datetime import datetime

import pandas as pd
import numpy as np
from sklearn import preprocessing

from cat_encoders import CatEncoder
from text_features_extractor import TextFeatureExtractor
from mailbox_feature_extractor import MailBoxFeatures


class Preprocessor:
    def __init__(self):
        self.mail_box_encoder = CatEncoder(preprocessing.OrdinalEncoder)
        self.contact_encoder = CatEncoder(preprocessing.OrdinalEncoder)
        self.timezone_encoder = CatEncoder(preprocessing.OrdinalEncoder)
        self.text_extractor = TextFeatureExtractor(feature_num=500)
        self.mail_box_features = MailBoxFeatures()

    @staticmethod
    def extract_time(time_str: str):
        if time_str in (None, 'none'):
            return 0, 0, 0, 0, 0

        dt = datetime.strptime(time_str, '%m/%d/%y %H:%M')
        return dt.year, dt.month, dt.day, dt.hour, dt.minute

    @staticmethod
    def series_to_numpy(ser: pd.Series):
        return np.array(ser).reshape(-1, 1)

    def process_datetime_series(self, dt: pd.Series):
        years, months, days, hours, minutes = [], [], [], [], []
        for s in dt:
            year, month, day, hour, minute = self.extract_time(s)
            years.append(int(year))
            months.append(int(month))
            days.append(int(day))
            hours.append(int(hour))
            minutes.append(int(minute))

        return np.array([years, months, days, hours, minutes]).T

    def fit_encoders(self, data: pd.DataFrame):
        self.mail_box_encoder.fit(self.series_to_numpy(data.MailBoxID))
        self.contact_encoder.fit(self.series_to_numpy(data.ContactID))
        self.timezone_encoder.fit(self.series_to_numpy(data.TimeZone))
        self.text_extractor.fit_tfidf_vectorization(data.Subject)
        self.mail_box_features.fit(data)

    def transform(self, data):
        mailbox_encoded = self.mail_box_encoder(self.series_to_numpy(data.MailBoxID))
        contact_encoded = self.contact_encoder(self.series_to_numpy(data.ContactID))
        timezone_encoded = self.timezone_encoder(self.series_to_numpy(data.TimeZone))
        time_matrix = self.process_datetime_series(data.SentOn)
        text_features = self.text_extractor(data.Subject)
        mailbox_features = self.mail_box_features.transform(data)
        return np.array(np.hstack([mailbox_encoded, contact_encoded, time_matrix,
                                   timezone_encoded, text_features, mailbox_features]),
                        dtype=float)


if __name__ == '__main__':
    df = pd.read_csv('../data/email_best_send_time_train.csv')
    df = df.fillna('none')
    print(df.head())

    p = Preprocessor()
    p.fit_encoders(df)
    transformed = p.transform(df)
    print(transformed.shape)
