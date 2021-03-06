import pandas as pd
import numpy as np
from sklearn import preprocessing

from cat_encoders import CatEncoder
from text_features_extractor import TextFeatureExtractor
from mailbox_feature_extractor import MailBoxFeatures
from time_features_extraction import TimeFeaturesExtractor
from recommender_features import RecommenderFeatures
from timezone_feature_extractor import TimeZoneFeatures
from contact_feature_extractor import ClientFeatures
from hacker_features import HackerFeatures


class Preprocessor:
    def __init__(self, text_feat_num: int = 500, text_feat_ngrams: tuple = (1, 2)):
        self.mail_box_encoder = CatEncoder(preprocessing.OrdinalEncoder)
        self.contact_encoder = CatEncoder(preprocessing.OrdinalEncoder)
        self.timezone_encoder = CatEncoder(preprocessing.OrdinalEncoder)
        self.text_extractor = TextFeatureExtractor(feature_num=500)
        self.mail_box_features = MailBoxFeatures()
        self.time_extractor = TimeFeaturesExtractor()
        self.recommender_extractor = RecommenderFeatures()
        self.timezone_features = TimeZoneFeatures()
        self.client_feature = ClientFeatures()
        self.hacker_features = HackerFeatures()

    @staticmethod
    def series_to_numpy(ser: pd.Series):
        return np.array(ser).reshape(-1, 1)

    def fit_encoders(self, data: pd.DataFrame):
        self.mail_box_encoder.fit(self.series_to_numpy(data.MailBoxID))
        self.contact_encoder.fit(self.series_to_numpy(data.ContactID))
        self.timezone_encoder.fit(self.series_to_numpy(data.TimeZone))
        self.text_extractor.fit_tfidf_vectorization(data.Subject)
        self.mail_box_features.fit(data)
        self.recommender_extractor.fit(data)
        self.timezone_features.fit(data)
        self.client_feature.fit(data)

    def transform(self, data):
        mailbox_encoded = self.mail_box_encoder(self.series_to_numpy(data.MailBoxID))
        contact_encoded = self.contact_encoder(self.series_to_numpy(data.ContactID))
        timezone_encoded = self.timezone_encoder(self.series_to_numpy(data.TimeZone))
        time_matrix = self.time_extractor.process_datetime_series(data.SentOn)
        aligned_time_matrix = self.time_extractor.process_aligned_datetime_series(data.SentOn, data.TimeZone)
        text_features = self.text_extractor(data.Subject)
        mailbox_features = self.mail_box_features.transform(data)
        recommender_features = self.recommender_extractor.transform(data)
        timezone_features = self.timezone_features.transform(data)
        client_feature = self.client_feature.transform(data)
        hacker_features = self.hacker_features.transform(data)
        return np.array(np.hstack([mailbox_encoded, contact_encoded, time_matrix, aligned_time_matrix,
                                   timezone_encoded, text_features, mailbox_features, recommender_features,
                                   timezone_features, client_feature, hacker_features]),
                        dtype=float)


if __name__ == '__main__':
    df = pd.read_csv('../data/new/email_best_send_time_train.csv')
    df = df.fillna('none')
    print(df.head())

    p = Preprocessor()
    p.fit_encoders(df)
    transformed = p.transform(df)
    print(transformed.shape)
