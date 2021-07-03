import numpy as np
import pandas as pd
from sklearn import preprocessing


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
