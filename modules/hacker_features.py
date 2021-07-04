class HackerFeatures:

    def transform(self, df):
        return df[['letter_number', 'letter_number_by_mail', 'letter_number_unequality']].values
