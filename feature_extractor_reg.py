import numpy as np

label_names = np.array(['A', 'B', 'Q', 'R'])


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        X_array = np.array([np.array(dd) for dd in X_df['spectra']])
        #X_array -= np.median(X_array, axis=1)[:, None]
        #X_array /= np.sqrt(np.sum(X_array ** 2, axis=1))[:, None]
        X_array = np.concatenate([X_array, X_df[label_names].values], axis=1)
        return X_array
