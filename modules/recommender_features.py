import pandas as pd
import numpy as np

from recommender_implicit import *


class RecommenderFeatures:
    def __init__(self, ):
        self.ContactID_features = {}
        self.MailBoxID_features = {}

    def fit(self, df):
        df_movies = df[["ContactID", "MailBoxID"]]
        df_movies.columns = ['user', 'item']
        df_movies["rating"] = df.Opened
        df_movies = df_movies.drop_duplicates()

        rec = Implicit()
        rec.set_k = 20

        test = pd.DataFrame({"user": df["ContactID"].values,
                             "item": df["MailBoxID"].values,
                             "rating": df["Opened"].values})

        rec.fit_trainset(df_movies)
        rec.fit_testset(test)
        params = {'factors': 40, 'regularization': 0.01, 'iterations': 10, 'use_native': True, 'use_cg': True,
                  'use_gpu': False, 'calculate_training_loss': True, 'num_threads': 4}
        rec.fit_model(params)

        self.ContactID_features[-1] = rec.model.user_factors.mean(axis=0)
        for k, v in rec.mapping_dict['user'].items():
            try:
                self.ContactID_features[k] = rec.model.user_factors[v]
            except:
                self.ContactID_features[k] = self.ContactID_features[-1]

        self.MailBoxID_features[-1] = rec.model.item_factors.mean(axis=0)
        for k, v in rec.mapping_dict['item'].items():
            try:
                self.MailBoxID_features[v] = rec.model.item_factors[k - 1]
            except:
                self.MailBoxID_features[v] = self.MailBoxID_features[-1]

    def transform(self, df):
        user_features = []
        item_features = []

        for u, i in zip(df.ContactID.values, df.MailBoxID.values):
            user_features.append(self.ContactID_features.get(u, self.ContactID_features[-1]))
            item_features.append(self.MailBoxID_features.get(i, self.MailBoxID_features[-1]))

        all_features = np.hstack([user_features, item_features])
        return all_features