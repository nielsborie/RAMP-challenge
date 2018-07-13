import numpy as np
import pandas as pd
from copy import copy
from xgboost import XGBRegressor
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFwe, f_regression, SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.linear_model import LassoLarsCV, RidgeCV,ElasticNetCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, RobustScaler, StandardScaler,MaxAbsScaler,FunctionTransformer
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from collections import Counter

class LogTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,num):
        self.num = num
    def fit(self,X, y=None, **kwargs):
        return self
    def transform(self, X, **kwargs):
        return np.log(X+self.num)
      
      
def correction(y_pred,possible_value,correction_threshold):
  y_pred = y_pred
  distances = np.abs(y_pred.reshape(-1, 1) - possible_value.reshape(1, -1))
  y_exact = possible_value[np.argmin(distances, axis=1)]
  y_pred_correct = np.where(abs(y_exact - y_pred) / y_exact < correction_threshold, y_exact, y_pred)
  return y_pred_correct

def voting_corrector(y1,y2,y3,y5,y6):
  l=[]
  l=[Counter(i).most_common()[0][0] for i in np.c_[y1.reshape(-1, 1),y2.reshape(-1, 1),y3.reshape(-1, 1),y5.reshape(-1, 1),y6.reshape(-1, 1)].tolist()]
  return np.asarray(l)


class StackingEstimator(BaseEstimator, TransformerMixin):
    """Meta-transformer for adding predictions and/or class probabilities as synthetic feature(s).
    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    """

    def __init__(self, estimator):
        """Create a StackingEstimator object.
        Parameters
        ----------
        estimator: object with fit, predict, and predict_proba methods.
            The estimator to generate synthetic features from.
        """
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        """Fit the StackingEstimator meta-transformer.
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like, shape (n_samples,)
            The target values (integers that correspond to classes in classification, real numbers in regression).
        fit_params:
            Other estimator-specific parameters.
        Returns
        -------
        self: object
            Returns a copy of the estimator
        """
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        """Transform data by adding two synthetic feature(s).
        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.
        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features + 1) or (n_samples, n_features + 1 + n_classes) for classifier with predict_proba attribute
            The transformed feature set.
        """
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

class ZeroCount(BaseEstimator, TransformerMixin):
    """Adds the count of zeros and count of non-zeros per sample as features."""

    def fit(self, X, y=None):
        """Dummy function to fit in with the sklearn API."""
        return self

    def transform(self, X, y=None):
        """Transform data by adding two virtual features.
        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components
            is the number of components.
        y: None
            Unused
        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features)
            The transformed feature set
        """
        X = check_array(X)
        n_features = X.shape[1]

        X_transformed = np.copy(X)

        non_zero_vector = np.count_nonzero(X_transformed, axis=1)
        non_zero = np.reshape(non_zero_vector, (-1, 1))
        zero_col = np.reshape(n_features - non_zero_vector, (-1, 1))

        X_transformed = np.hstack((non_zero, X_transformed))
        X_transformed = np.hstack((zero_col, X_transformed))

        return X_transformed

class Regressor(BaseEstimator):
    def __init__(self):
        self.list_molecule = ['A', 'B', 'Q', 'R']
        self.possible_value = {}
        self.possible_value["A"] = np.array([1000, 5000, 2000, 1400, 800, 400, 1600, 600, 300, 10000])
        self.possible_value["B"] = np.array([500, 20000, 25000, 10000, 2000, 5000, 4000, 7000, 1500, 1000])
        self.possible_value["Q"] = np.array([10000, 1000, 2000, 7000, 3000, 6000, 8000, 4000, 5000, 9000])
        self.possible_value["R"] = np.array([800, 4000, 1600, 10000, 2000, 400, 1000, 3000, 5000, 1200])
        self.possible_values = np.array([800,1600,20000,4000,8000,1000,5000,25000,600,300,9000,400,2000,10000,7000,500,6000,3000,1200,1400,1500])
        self.dict_reg1 = {}
        self.dict_reg2 = {}
        self.list_reg1 = [
          make_pipeline(
            make_union(
              StackingEstimator(estimator=make_pipeline(
                  StackingEstimator(estimator=LinearSVR(C=25.0, dual=True, epsilon=1.0, loss="epsilon_insensitive", tol=0.001)),
                  StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=8, min_child_weight=20, n_estimators=100, nthread=1, subsample=1.0)),
                  PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
                  LassoLarsCV(normalize=True)
                )),
              FunctionTransformer(copy)
            ),
            DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, min_samples_split=15)
          ),
          make_pipeline(
            make_union(
              make_pipeline(
                StackingEstimator(estimator=LassoLarsCV(normalize=False)),
            	PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            	RobustScaler(),
            	StandardScaler()
              ),
              FunctionTransformer(copy)
            ),
            StackingEstimator(estimator=LassoLarsCV(normalize=True)),
            SelectFwe(score_func=f_regression, alpha=0.045),
            XGBRegressor(learning_rate=0.1, max_depth=10, min_child_weight=1, n_estimators=100, nthread=1, subsample=1.0)
          ),
          make_pipeline(
            make_union(
              StackingEstimator(estimator=LassoLarsCV(normalize=True)),
              StackingEstimator(estimator=make_pipeline(
                  RobustScaler(),
                  ZeroCount(),
                  MinMaxScaler(),
                  PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
                  RidgeCV()
                ))
            ),
            DecisionTreeRegressor(max_depth=9, min_samples_leaf=2, min_samples_split=14)
          ),
          make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            StackingEstimator(estimator=LassoLarsCV(normalize=False)),
            StackingEstimator(estimator=DecisionTreeRegressor(max_depth=7, min_samples_leaf=10, min_samples_split=19)),
            StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=2, min_child_weight=6, n_estimators=100, nthread=1, subsample=0.8500000000000001)),
            DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, min_samples_split=15)
          )
          ]
        
        self.list_reg2 = [
          make_pipeline(
            MinMaxScaler(),
            StackingEstimator(estimator=RidgeCV()),
            StackingEstimator(estimator=XGBRegressor(learning_rate=0.01, max_depth=9, min_child_weight=14, n_estimators=100, nthread=1, subsample=0.7500000000000001)),
            StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.7000000000000001, min_samples_leaf=18, min_samples_split=5, n_estimators=100)),
            StackingEstimator(estimator=RidgeCV()),
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            StackingEstimator(estimator=LinearSVR(C=0.5, dual=True, epsilon=0.0001, loss="epsilon_insensitive", tol=0.0001)),
            LassoLarsCV(normalize=True)
          ),
          make_pipeline(
            StackingEstimator(estimator=ElasticNetCV(l1_ratio=1.0, tol=0.001)),
            StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.35000000000000003, min_samples_leaf=13, min_samples_split=13, n_estimators=100)),
            StandardScaler(),
            StackingEstimator(estimator=DecisionTreeRegressor(max_depth=7, min_samples_leaf=16, min_samples_split=2)),
            MaxAbsScaler(),
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            StackingEstimator(estimator=RidgeCV()),
            XGBRegressor(learning_rate=0.5, max_depth=8, min_child_weight=2, n_estimators=100, nthread=1, subsample=0.9500000000000001)
          ),
          make_pipeline(
            StackingEstimator(estimator=LinearSVR(C=25.0, dual=True, epsilon=1.0, loss="squared_epsilon_insensitive", tol=1e-05)),
            StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=14, p=2, weights="distance")),
            StackingEstimator(estimator=XGBRegressor(learning_rate=0.01, max_depth=1, min_child_weight=3, n_estimators=100, nthread=1, subsample=0.3)),
            DecisionTreeRegressor(max_depth=10, min_samples_leaf=6, min_samples_split=5)
          ),
          make_pipeline(
            make_union(
              StackingEstimator(estimator=make_pipeline(
                  make_union(
                    make_pipeline(
                      MinMaxScaler(),
                      PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
                      MinMaxScaler(),
                      StandardScaler(),
                      MinMaxScaler(),
                      ZeroCount(),
                      StackingEstimator(estimator=RidgeCV()),
                      SelectFromModel(estimator=ExtraTreesRegressor(max_features=0.45, n_estimators=100), threshold=0.0)
                    ),
                    MinMaxScaler()
                  ),
                  KNeighborsRegressor(n_neighbors=6, p=1, weights="distance")
                )),
              FunctionTransformer(copy)
            ),
            XGBRegressor(learning_rate=0.1, max_depth=9, min_child_weight=3, n_estimators=100, nthread=1, subsample=0.9000000000000001)
          )

        ]
        self.list_reg5 = [make_pipeline(
            StackingEstimator(estimator=RidgeCV()),
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            StackingEstimator(estimator=LassoLarsCV(normalize=False)),
            SelectFwe(score_func=f_regression, alpha=0.049),
            StackingEstimator(estimator=XGBRegressor(learning_rate=0.5, max_depth=8, min_child_weight=2, n_estimators=100, nthread=1, subsample=0.45)),
            DecisionTreeRegressor(max_depth=6, min_samples_leaf=7, min_samples_split=14)
          ),
                          make_pipeline(
            ZeroCount(),
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            StackingEstimator(estimator=XGBRegressor(learning_rate=0.001, max_depth=4, min_child_weight=7, n_estimators=100, nthread=1, subsample=0.3)),
            StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.1, loss="quantile", max_depth=3, max_features=0.2, min_samples_leaf=20, min_samples_split=6, n_estimators=100, subsample=0.9000000000000001)),
            StackingEstimator(estimator=DecisionTreeRegressor(max_depth=1, min_samples_leaf=2, min_samples_split=15)),
            StackingEstimator(estimator=LassoLarsCV(normalize=False)),
            MaxAbsScaler(),
            StackingEstimator(estimator=XGBRegressor(learning_rate=1.0, max_depth=8, min_child_weight=1, n_estimators=100, nthread=1, subsample=0.8500000000000001)),
            XGBRegressor(learning_rate=0.1, max_depth=4, min_child_weight=1, n_estimators=100, nthread=1, subsample=0.9500000000000001)
          ),
                          make_pipeline(
            StackingEstimator(estimator=LassoLarsCV(normalize=True)),
            StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=41, p=2, weights="distance")),
            ZeroCount(),
            DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, min_samples_split=3)
          ),
                          make_pipeline(
            StackingEstimator(estimator=RidgeCV()),
            StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=10, min_child_weight=3, n_estimators=100, nthread=1, subsample=0.8)),
            StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.7500000000000001, min_samples_leaf=20, min_samples_split=3, n_estimators=100)),
            StackingEstimator(estimator=DecisionTreeRegressor(max_depth=8, min_samples_leaf=1, min_samples_split=2)),
            DecisionTreeRegressor(max_depth=8, min_samples_leaf=2, min_samples_split=6)
          )]
        
        self.dict_reg3 = {}
        self.list_compo = [11,11,10,10]
        self.list_C = [100000,100000,1000,100000]
        self.list_gamma = [0.001,0.001,0.0001,0.0001]
        self.list_reducer = [FactorAnalysis(copy=True, 
                                            iterated_power=3, 
                                            max_iter=1000, 
                                            n_components=11,
                                            noise_variance_init=None, 
                                            random_state=0, 
                                            svd_method='randomized',
                                            tol=0.01),
                             FactorAnalysis(copy=True, 
                                            iterated_power=3, 
                                            max_iter=1000,
                                            n_components=11,
                                            noise_variance_init=None, 
                                            random_state=0, 
                                            svd_method='randomized',
                                            tol=0.01),
                             PCA(copy=True, 
                                 iterated_power=5, 
                                 n_components=10, 
                                 random_state=None,
                                 svd_solver='auto', 
                                 tol=0.0, 
                                 whiten=False),
                             FactorAnalysis(copy=True, 
                                            iterated_power=3, 
                                            max_iter=1000, 
                                            n_components=10,
                                            noise_variance_init=None, 
                                            random_state=0, 
                                            svd_method='randomized',
                                            tol=0.01)]
        self.dict_reg4 = {}
        self.dict_reg5 = {}
        
        for i,mol in enumerate(self.list_molecule):
            self.dict_reg1[mol] = Pipeline([
                ('standard', StandardScaler()),
                ('reduce_dim', FactorAnalysis(copy=True, 
                                            iterated_power=3, 
                                            max_iter=1000, 
                                            n_components=11,
                                            noise_variance_init=None, 
                                            random_state=0, 
                                            svd_method='randomized',
                                            tol=0.01)),
                ('reg', self.list_reg1[i]),
            ])
            self.dict_reg2[mol] = Pipeline([
                ('extract', LogTransformer(num=1)), 
                ('reduce_dim', FactorAnalysis(n_components=10,random_state=12)),
                ('reg', self.list_reg2[i]),
            ])
            self.dict_reg3[mol] = Pipeline([
                ('standard', StandardScaler()),
                ('pca', self.list_reducer[i]),
                ('reg', SVC(C=self.list_C[i], gamma=self.list_gamma[i], kernel='poly', degree=2, coef0=2, probability=True))
            ])
            self.dict_reg4[mol] = RidgeCV(cv=10)
            self.dict_reg5[mol] = Pipeline([
                ('extract', LogTransformer(num=2)), 
                ('reduce_dim', FactorAnalysis(n_components=12,random_state=11)),
                ('reg', self.list_reg5[i]),
            ])

    def fit(self, X, y):
        for i, mol in enumerate(self.list_molecule):
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]
            X_mol = X[ind_mol]
            y_mol = y[ind_mol]
            self.dict_reg1[mol].fit(X_mol, y_mol)
            self.dict_reg2[mol].fit(X_mol, y_mol)
            self.dict_reg3[mol].fit(X_mol, y_mol)
            self.dict_reg4[mol].fit(X_mol, y_mol)
            self.dict_reg5[mol].fit(X_mol, y_mol)

    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        y_pred1 = np.zeros(X.shape[0])
        y_pred2 = np.zeros(X.shape[0])
        y_pred3 = np.zeros(X.shape[0])
        y_pred4 = np.zeros(X.shape[0])
        y_pred5 = np.zeros(X.shape[0])
        y_pred6 = np.zeros(X.shape[0])
        
        for i, mol in enumerate(self.list_molecule):
            ind_mol = np.where(np.argmax(X[:, -4:], axis=1) == i)[0]
            X_mol = X[ind_mol]
            #y_pred1[ind_mol] = correction(self.dict_reg1[mol].predict(X_mol),self.possible_value[mol],0.2)
            y_pred1[ind_mol] = correction(self.dict_reg1[mol].predict(X_mol),self.possible_value[mol],0.5)
            y_pred2[ind_mol] = correction(self.dict_reg2[mol].predict(X_mol),self.possible_value[mol],0.5)
            y_pred3[ind_mol] = correction(self.dict_reg3[mol].predict(X_mol),self.possible_value[mol],0.5)
            y_pred4[ind_mol] = correction(self.dict_reg4[mol].predict(X_mol),self.possible_value[mol],0.5)
            y_pred5[ind_mol] = correction(self.dict_reg5[mol].predict(X_mol),self.possible_value[mol],0.5)
            #y_pred[ind_mol] = correction((1/3)*(y_pred1[ind_mol]+y_pred2[ind_mol]+y_pred3[ind_mol]),self.possible_value[mol],0.2)
            #y_pred[ind_mol] = correction(y_pred5[ind_mol],self.possible_value[mol],0.2)
            y_pred6[ind_mol] = correction((1/4)*(y_pred1[ind_mol]+y_pred2[ind_mol]+y_pred3[ind_mol]+y_pred5[ind_mol]),self.possible_value[mol],0.5)
            y_pred[ind_mol] = voting_corrector(y_pred1[ind_mol],y_pred2[ind_mol],y_pred3[ind_mol],y_pred5[ind_mol],y_pred6[ind_mol])
        return y_pred

