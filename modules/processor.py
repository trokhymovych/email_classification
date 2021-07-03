from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing


class TextFeatureExtractor:
    def __init__(self, feature_num: int = 500):
        self.feature_num = feature_num
        self.tf_idf_vectorizer = None

    def fit_tfidf_vectorization(self, texts: pd.Series):
        self.tf_idf_vectorizer = TfidfVectorizer(max_features=self.feature_num)
        self.tf_idf_vectorizer.fit(texts.astype(str))

    @staticmethod
    def remove_brackets(txt):
        if txt.startswith('"'):
            txt = txt[1:]
        if txt.endswith('"'):
            txt = txt[:-1]
        return txt

    def text_starts_with_re(self, texts: pd.Series):
        res = []
        for text in texts:
            if self.remove_brackets(text).lower().startswith('re'):
                res.append(1)
            else:
                res.append(0)
        return np.array(res)

    def __call__(self, texts: pd.Series):
        assert self.tf_idf_vectorizer is not None
        vectorized_texts = self.tf_idf_vectorizer.fit_transform(texts.astype(str)).todense()
        text_is_reply = self.text_starts_with_re(texts).reshape(-1, 1)
        return np.array(np.hstack([text_is_reply, vectorized_texts]),
                        dtype=float)


class CatEncoder:
    def __init__(self, encoder_type):
        self.encoder_type = encoder_type
        self.encoder = None

    def fit(self, data: pd.Series) -> None:
        if self.encoder_type == preprocessing.OrdinalEncoder:
            self.encoder = self.encoder_type(handle_unknown='use_encoded_value', unknown_value=np.nan)
        elif self.encoder_type == preprocessing.OneHotEncoder:
            self.encoder = self.encoder_type(handle_unknown='ignore')
        else:
            self.encoder = self.encoder_type()

        self.encoder.fit(data)

    def __call__(self, data: pd.Series) -> np.ndarray:
        assert self.encoder is not None

        return self.encoder.transform(data)


class Preprocessor:
    def __init__(self):
        self.mail_box_encoder = CatEncoder(preprocessing.OrdinalEncoder)
        self.contact_encoder = CatEncoder(preprocessing.OrdinalEncoder)
        self.timezone_encoder = CatEncoder(preprocessing.OrdinalEncoder)
        self.text_extractor = TextFeatureExtractor(feature_num=500)

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

    def transform(self, data):
        mailbox_encoded = self.mail_box_encoder(self.series_to_numpy(data.MailBoxID))
        contact_encoded = self.contact_encoder(self.series_to_numpy(data.ContactID))
        timezone_encoded = self.timezone_encoder(self.series_to_numpy(data.TimeZone))
        time_matrix = self.process_datetime_series(data.SentOn)
        text_features = self.text_extractor(data.Subject)
        return np.array(np.hstack([mailbox_encoded, contact_encoded, time_matrix, timezone_encoded, text_features]),
                        dtype=float)


if __name__ == '__main__':
    df = pd.read_csv('../data/email_best_send_time_train.csv')
    df = df.fillna('none')
    print(df.head())

    p = Preprocessor()
    p.fit_encoders(df)
    transformed = p.transform(df)
    print(transformed.shape)
