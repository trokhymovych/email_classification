import re
import string

from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureExtractor:
    def __init__(self, feature_num: int = 500):
        self.feature_num = feature_num
        self.tf_idf_vectorizer = None

        with open('text_data_utils/popular_mail_subjects.txt', 'r') as f:
            lines = f.readlines()
        self.popular_subjects = [ln.replace("\n", "") for ln in lines]

        with open('text_data_utils/names.txt', 'r') as f:
            lines = f.readlines()
        self.names = set([ln.replace("\n", "") for ln in lines])

        self.regex = re.compile('[%s]' % re.escape(string.punctuation))
        self.lemm = WordNetLemmatizer()


    @staticmethod
    def remove_brackets(txt):
        if txt.startswith('"'):
            txt = txt[1:]
        if txt.endswith('"'):
            txt = txt[:-1]
        return txt

    def remove_punctuation(self, s):
        return self.regex.sub('', s)

    def replace_names_with_token(self, texts: pd.Series, token: str = 'name'):
        bool_res = []
        res = []
        for text in texts:
            text = self.remove_punctuation(text)
            intersection = set(text.split(' ')).intersection(self.names)
            if len(intersection):
                for isect in intersection:
                    text = text.replace(isect, token)
                bool_res.append(1)
            else:
                bool_res.append(0)

            res.append(' '.join([self.lemm.lemmatize(t, pos='a') for t in text.lower().split()]))
        return np.array(bool_res).reshape(-1, 1), np.array(res)

    def fit_tfidf_vectorization(self, texts: pd.Series):
        _, texts = self.replace_names_with_token(texts)
        self.tf_idf_vectorizer = TfidfVectorizer(max_features=self.feature_num, ngram_range=(1, 2))
        self.tf_idf_vectorizer.fit(texts.astype(str))

    def text_starts_with_re(self, texts: pd.Series):
        res = []
        for text in texts:
            if self.remove_brackets(text).lower().startswith('re'):
                res.append(1)
            else:
                res.append(0)
        return np.array(res).reshape(-1, 1)

    @staticmethod
    def text_contains_pattern(texts: pd.Series, expr: str) -> np.ndarray:
        res = []
        for text in texts:
            if len(re.findall(expr, str(text))):
                res.append(1)
            else:
                res.append(0)
        return np.array(res).reshape(-1, 1)

    @staticmethod
    def text_perfect_match(texts: pd.Series, exact_pattern: str) -> np.ndarray:
        res = []
        for text in texts:
            if text == exact_pattern:
                res.append(1)
            else:
                res.append(0)
        return np.array(res).reshape(-1, 1)

    def __call__(self, texts: pd.Series):
        assert self.tf_idf_vectorizer is not None

        text_is_reply = self.text_starts_with_re(texts)

        text_features = [text_is_reply, ]

        for pattern in ('Hi.*!', 'RE: Hi.*!', '.*[?].*'):
            text_features.append(self.text_contains_pattern(texts, pattern))

        # WARNING: exact match needed - all preprocessing afterwards!
        for pattern in self.popular_subjects:
            text_features.append(self.text_perfect_match(texts, pattern))

        contains_name, updated_texts = self.replace_names_with_token(texts)

        vectorized_texts = self.tf_idf_vectorizer.transform(updated_texts.astype(str)).todense()
        text_features.append(contains_name)
        text_features.append(vectorized_texts)

        return np.array(np.hstack(text_features), dtype=float)


if __name__ == '__main__':
    text_feature_extractor = TextFeatureExtractor()
    print(text_feature_extractor)
