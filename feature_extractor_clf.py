import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.signal import periodogram

class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        """
        X_array = np.array([np.array(dd) for dd in X_df['spectra']])
        f, Pxx_den = periodogram(X_array)
        self.selection = SelectKBest(k=50)
        self.selection.fit(periodogram(X_array)[1],y)
        """
        pass

    def transform(self, X_df):
        X_array = np.array([np.array(dd) for dd in X_df['spectra']])
        #pca = PCA(n_components=10)
        # Maybe some original features where good, too?
        #selection = SelectKBest(k=10)

        # Build estimator from PCA and Univariate selection:
        #combined_features = FeatureUnion([("pca", pca), ("select", selection)])

        # Use combined features to transform dataset:
        #X_features = combined_features.fit(X_array, y_df.molecule).transform(X_array)
        #f, Pxx_den = periodogram(X_array)
        #X_features = self.selection.transform(Pxx_den)
        
        """
        return np.c_[X_features,
                     np.mean(X_array,axis=1),
                     np.max(X_array,axis=1),
                     np.min(X_array,axis=1),
                     np.median(X_array,axis=1),
                     np.std(X_array,axis=1)]
        """
        return X_array