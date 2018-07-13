import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA, FactorAnalysis,KernelPCA,NMF
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer,RobustScaler,MinMaxScaler,MaxAbsScaler, Normalizer
from sklearn.linear_model import LassoLarsCV, RidgeCV, LogisticRegression,LogisticRegressionCV
from sklearn.pipeline import make_pipeline, make_union,FeatureUnion,Pipeline
from sklearn.svm import LinearSVR,SVC
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor,XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.feature_selection import RFE, SelectFwe, f_classif,SelectFwe, f_regression,SelectPercentile
from copy import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
from sklearn.feature_selection import SelectKBest
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import FastICA
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import StratifiedKFold

class Blending_avg(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator
        self.n_folds = 10
        self.verbose = True
        self.shuffle = False

    def fit(self, X, y):
        n_classes = len(set(y))
        #self.output = np.zeros((X.shape[0], n_classes))
        skf = StratifiedKFold(y, self.n_folds)
        self.D = {}
        for i,(tra, tst) in enumerate(skf):
            self.D[i] = self.estimator.fit(X[tra], y[tra])
        return self

    def transform(self, X,y=None):
        A = np.zeros((X.shape[0], 1))
        B = np.zeros((X.shape[0], 1))
        Q = np.zeros((X.shape[0], 1))
        R = np.zeros((X.shape[0], 1))
        for key,val in self.D.items():
            split = self.D[key].predict_proba(X)
            A = np.c_[A,split[:,0]]
            B = np.c_[B,split[:,1]]
            Q = np.c_[Q,split[:,2]]
            R = np.c_[R,split[:,3]]
        A = np.mean(np.delete(A, 0, 1), axis=1)
        B = np.mean(np.delete(B, 0, 1), axis=1)
        Q = np.mean(np.delete(Q, 0, 1), axis=1)
        R = np.mean(np.delete(R, 0, 1), axis=1)
        return np.c_[A,B,Q,R]
class Blending(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator
        self.n_folds = 5
        self.verbose = True
        self.shuffle = False

    def fit(self, X, y):
        n_classes = len(set(y))
        #self.output = np.zeros((X.shape[0], n_classes))
        skf = StratifiedKFold(y, self.n_folds)
        self.D = {}
        for i,(tra, tst) in enumerate(skf):
            self.D[i] = self.estimator.fit(X[tra], y[tra])
        return self

    def transform(self, X,y=None):
        X_transformed = np.zeros((X.shape[0], 1))
        for key,val in self.D.items():
            self.D[key].predict_proba(X)
            X_transformed=np.c_[X_transformed,self.D[key].predict_proba(X)]
        return np.delete(X_transformed, 0, 1)

class LogTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,num):
        self.num = num
    def fit(self,X, y=None, **kwargs):
        return self
    def transform(self, X, **kwargs):
        return np.log(X+self.num)
      
class ModelClassTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self
    def transform(self, X, **transform_params):
        return self.model.predict_proba(X)
      
      
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

        return np.hstack((self.estimator.predict_proba(X), X))
      
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

class Classifier(BaseEstimator):
    def __init__(self):
      self.model = {}
      self.model[0] = Pipeline([
          ('extract', LogTransformer(num=1)),
          ('pca', FactorAnalysis(copy=True, 
                                 iterated_power=3, 
                                 max_iter=1000, 
                                 n_components=10,
                                 noise_variance_init=None, 
                                 random_state=75, 
                                 svd_method='randomized',
                                 tol=0.01)),
          ('clf', make_pipeline(
              StandardScaler(),
              MLPClassifier(activation="tanh", 
                            alpha=1.4677992676220734e-12, 
                            epsilon=1e-12, 
                            hidden_layer_sizes=(131, 132, 91, 50, 20, 10), 
                            solver="adam", 
                            warm_start=True)
            ))
        ])
      self.model[1] =  Pipeline(memory=None,
                                steps=[('extract', LogTransformer(num=0.6812920690579611)), 
                                       ('reduce_dim', FactorAnalysis(copy=True, 
                                                                     iterated_power=3, 
                                                                     max_iter=1000, 
                                                                     n_components=10,
                                                                     noise_variance_init=None, 
                                                                     random_state=0, 
                                                                     svd_method='randomized',
                                                                     tol=0.01)), 
                                       ('mlp', MLPClassifier(hidden_layer_sizes= (50, 25, 60, 4, 8, 10),
                                                             solver="adam",
                                                             activation="tanh",
                                                             batch_size=30,
                                                             epsilon=1e-12))
                                      ])
      self.model[2] = Pipeline([
          ('extract', LogTransformer(num=0.6)), 
          ('pca', FactorAnalysis(n_components=10,random_state=10)),
          ('clf', make_pipeline(
              make_union(
                FunctionTransformer(copy),
                StackingEstimator(estimator=MLPClassifier(activation="tanh", alpha=5.623413251903491e-09, epsilon=5.6234132519034906e-14, hidden_layer_sizes=(100, 100, 75, 30, 10), solver="adam", warm_start=True))
              ),
              StackingEstimator(estimator=MLPClassifier(activation="tanh", alpha=31622776.60168379, epsilon=3.162277660168379e-13, hidden_layer_sizes=(100, 100, 75, 30, 10), solver="adam", warm_start=True)),
              MLPClassifier(activation="tanh", alpha=1e-09, epsilon=1.7782794100389228e-12, hidden_layer_sizes=(50, 25, 60, 4, 8, 10), solver="adam", warm_start=True)
            )
          )
        ])
      self.model[3] = Pipeline([
          ('extract', LogTransformer(num=2)), 
          ('pca', FactorAnalysis(copy=True, 
                                 iterated_power=3, 
                                 max_iter=1000, 
                                 n_components=10,
                                 noise_variance_init=None, 
                                 random_state=None, 
                                 svd_method='randomized',
                                 tol=0.01)),
          ('clf', MLPClassifier(hidden_layer_sizes=(90, 90, 60, 30, 14, 6),
                                activation="tanh",
                                solver="adam",
                                warm_start = True,
                                epsilon=1e-14))
        ])
      self.model[4] = Pipeline([
          ('extract', LogTransformer(num=0.5)), 
          ('pca', KernelPCA(alpha=1.0, 
                            coef0=1, 
                            copy_X=True, 
                            degree=3, 
                            eigen_solver='auto',
                            fit_inverse_transform=False, 
                            gamma=None, 
                            kernel='linear',
                            kernel_params=None, 
                            max_iter=None, 
                            n_components=10, 
                            n_jobs=1,
                            random_state=None, 
                            remove_zero_eig=False, tol=0)),
          ('std',StandardScaler()),
          ('clf', MLPClassifier(hidden_layer_sizes=(50, 25, 60, 4, 8, 10),
                                activation="tanh",
                                solver="adam",
                                batch_size=50) )
        ])
      self.model[5] = Pipeline([
          ('extract', LogTransformer(num=0.5)), 
          ('pca', KernelPCA(alpha=0.0001, 
                            coef0=1, 
                            copy_X=True, 
                            degree=3, 
                            eigen_solver='auto',
                            fit_inverse_transform=False, 
                            gamma=None, 
                            kernel='linear',
                            kernel_params=None, 
                            max_iter=100, 
                            n_components=10, 
                            n_jobs=1,
                            random_state=19, 
                            remove_zero_eig=False, 
                            tol=0.001)),
          ('std',StandardScaler()),
          ('clf', MLPClassifier(hidden_layer_sizes=(112, 100, 77, 34, 22, 10),
                                activation="tanh",
                                solver="adam",
                                alpha=0.01,
                                shuffle = True,
                                warm_start = True,
                                batch_size=60,
                                early_stopping=False,
                                max_iter=100,
                                random_state = 140 ))
        ])
      self.model[6] = VotingClassifier(estimators=[('model1', self.model[1]), 
                                                   ('model2', self.model[2]),
                                                   ('model3', self.model[3]),
                                                   ('model4', self.model[4]),
                                                   ('model5', self.model[5])
                                                  ], 
                                       weights=[1,1,1,1,1],
                                       voting='soft')
      self.model[7] = Pipeline([
          ('extract', LogTransformer(num=1)), 
          ('pca', FactorAnalysis(n_components=10,random_state=40)),
          ('clf', make_pipeline(
              SelectFwe(score_func=f_classif, alpha=0.028),
              PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
              StandardScaler(),
              LogisticRegression(C=5.0, dual=False, penalty="l1")
            )
          )
        ])
      self.model[8] = Pipeline([
          ('extract', LogTransformer(num=0.5)), 
          ('pca', FactorAnalysis(n_components=10)),
          ('clf', SVC(C=1, 
                      cache_size=200, 
                      class_weight='balanced', 
                      coef0=5.0,
                      decision_function_shape='ovr', 
                      degree=5, gamma=0.01, 
                      kernel='poly',
                      max_iter=-1, 
                      probability=True, 
                      random_state=None, 
                      shrinking=True,
                      tol=0.001, verbose=False) )
        ])
      self.model[9] = Pipeline([
          ('extract', LogTransformer(num=0.5)), 
          ('pca', FactorAnalysis(n_components=10)),
          ('clf', LogisticRegression(C=5,
                                     class_weight="balanced",
                                     dual=False,
                                     fit_intercept=True,
                                     intercept_scaling=1,
                                     max_iter=25,
                                     multi_class="multinomial",
                                     penalty="l2",
                                     solver="lbfgs",
                                     warm_start=True,
                                     tol=0.0001,
                                     verbose=0,
                                     n_jobs=1,
                                     random_state=None)  )
        ])
      self.model[10] = Pipeline([
          ('extract', LogTransformer(num=1)), 
          ("xgb",XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                               colsample_bytree=0.7, gamma=0, learning_rate=0.2, max_delta_step=0,
                               max_depth=5, min_child_weight=0, missing=None, n_estimators=180,
                               n_jobs=1, nthread=4, objective='binary:logistic', random_state=0,
                               reg_alpha=1e-05, reg_lambda=1, scale_pos_weight=1, seed=27,
                               silent=True, subsample=0.7))

        ])
      self.model[11] = Pipeline([
          ('scaler', StandardScaler()),
          ('pca', FactorAnalysis(copy=True, 
                                 iterated_power=3, 
                                 max_iter=1000, 
                                 n_components=10,
                                 noise_variance_init=None, 
                                 random_state=0, 
                                 svd_method='randomized',
                                 tol=0.01)),
          ('clf', MLPClassifier(activation="tanh", 
                                alpha=0.01, 
                                epsilon=1e-11, 
                                hidden_layer_sizes=(85, 50, 60, 20, 30, 10), 
                                solver="adam", 
                                warm_start=False,
                                random_state=57))
        ])
      self.model[12] = make_pipeline(LogTransformer(0),
                                     PCA(n_components=10,random_state=10),
                                     ExtraTreesClassifier(n_estimators=150,random_state=63))
      self.model[13] = make_pipeline(LogTransformer(0),
                                     PCA(n_components=100,random_state=10),
                                     GradientBoostingClassifier(n_estimators=100))
      self.model[14] = make_pipeline(LogTransformer(0),
                                     PCA(n_components=100,random_state=10),
                                     XGBClassifier(max_depth=6,n_estimators=100))
      self.model[15] = make_pipeline(LogTransformer(0),
                                     PCA(n_components=100,random_state=10),
                                     RandomForestClassifier(n_estimators=100))
      self.model[16] = Pipeline([
          ('std',StandardScaler()),
          ('pca', FactorAnalysis(n_components=10)),
          ('clf', make_pipeline(
              SelectFwe(score_func=f_classif, alpha=0.025),
              ZeroCount(),
              PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
              RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.25, n_estimators=100), step=0.25),
              LogisticRegression(C=25.0, dual=False, penalty="l1")
            ))
        ])
      self.model[17] = Pipeline([
          ('extract', LogTransformer(num=0.5)), 
          ('pca', FactorAnalysis(n_components=10)),
          ('clf', XGBClassifier( learning_rate =0.2, n_estimators=260, max_depth=3,
                                min_child_weight=1, gamma=0, subsample=0.7, colsample_bytree=0.6,reg_alpha=1e-05,
                                objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27))
        ])
      self.model[18] = Pipeline(memory=None,
                                steps=[('extract', LogTransformer(num=0.6)), 
                                       ('reduce_dim', FactorAnalysis(copy=True, iterated_power=3, max_iter=1000, n_components=9,
                                                                     noise_variance_init=None, random_state=0, svd_method='randomized',
                                                                     tol=0.01)), 
                                       ('svm', SVC(C=177827.94100389228, cache_size=200, class_weight=None, coef0=0.0,
                                                   decision_function_shape='ovr', degree=3, gamma=0.0006812920690579623,
                                                   kernel='rbf', max_iter=-1, probability=True, random_state=None,
                                                   shrinking=True, tol=0.001, verbose=False))])
      self.model[19] = Pipeline(memory=None,
                                steps=[('extract', LogTransformer(num=0.6812920690579611)), 
                                       ('reduce_dim', FactorAnalysis(copy=True, 
                                                                     iterated_power=3, 
                                                                     max_iter=1000, 
                                                                     n_components=9,
                                                                     noise_variance_init=None, 
                                                                     random_state=0, 
                                                                     svd_method='randomized',
                                                                     tol=0.01)), 
                                       ('svm', SVC(C=1211527.6586285902, 
                                                   cache_size=200, 
                                                   class_weight=None, 
                                                   coef0=0.0,
                                                   decision_function_shape='ovr', 
                                                   degree=3, 
                                                   gamma=0.00026101572156825384,
                                                   kernel='rbf', 
                                                   max_iter=-1, 
                                                   probability=True, 
                                                   random_state=None,
                                                   shrinking=True, 
                                                   tol=0.001, 
                                                   verbose=False))
                                      ])
      self.model[20] = Pipeline([
          ('extract', LogTransformer(num=2)), 
          ('pca', FactorAnalysis(n_components=10, 
                                 tol=0.01, 
                                 copy=True, 
                                 max_iter=3000, 
                                 noise_variance_init=None, 
                                 svd_method='randomized', 
                                 iterated_power=3, 
                                 random_state=92)),
          ('clf', MLPClassifier(hidden_layer_sizes=(135, 135, 90, 45, 22, 10), 
                                random_state=42, 
                                max_iter=1000, 
                                warm_start=True, 
                                epsilon=1e-16, 
                                alpha=1e-6, 
                                activation='tanh'))
        ])
      self.model[21] = Pipeline([
          ('extract', LogTransformer(num=2)), 
          ('pca', FactorAnalysis(n_components=10, 
                                 tol=0.01, 
                                 copy=True, 
                                 max_iter=3000, 
                                 noise_variance_init=None, 
                                 svd_method='randomized', 
                                 iterated_power=3, 
                                 random_state=None)),
          ('clf', MLPClassifier(hidden_layer_sizes=(135, 135, 90, 45, 22, 10), 
                                random_state=None, 
                                max_iter=1000, 
                                warm_start=True, 
                                epsilon=1e-16, 
                                alpha=1e-6, 
                                activation='tanh'))
        ])
      self.model[22] = VotingClassifier(estimators=[('mlp20', self.model[20]), 
                                        ('mlp0', self.model[0]),
                                    ('VotingClassifier', self.model[6])
                                    ], 
                        weights=[1,0.5,0.5],
                        voting='soft') 
      
      self.model[23]  = make_pipeline(LogTransformer(num=2),
                                      MaxAbsScaler(),
                                      make_union(NMF(n_components=20, init='random', random_state=0),
                                                 KMeans(n_clusters=5, random_state=0),
                                                 Isomap(n_neighbors=5, n_components=3),
                                                 SelectKBest(k=10)
                                                ),
                                      ExtraTreesClassifier(bootstrap=False, 
                                                           criterion="gini",
                                                           max_features=0.8500000000000001, 
                                                           min_samples_leaf=3, 
                                                           min_samples_split=8, 
                                                           n_estimators=100))
      self.model[24] = make_pipeline(
        MaxAbsScaler(),
        PCA(n_components=100,random_state=0),
        AdaBoostClassifier(ExtraTreesClassifier(n_estimators=150,random_state=63),random_state=0)
      )
      self.model[25]  = make_pipeline(
        MaxAbsScaler(),
        PCA(n_components=100,random_state=0),
        BaggingClassifier(ExtraTreesClassifier(n_estimators=150,random_state=63),random_state=0)
      )
      self.model[26]  = make_pipeline(
        MaxAbsScaler(),
        PCA(n_components=100,random_state=0),
        BaggingClassifier(RandomForestClassifier(n_estimators=200,max_depth=10,random_state=10),random_state=0)
      )
      self.model[27] = make_pipeline(LogTransformer(num=0.4),
                                     FactorAnalysis(n_components=12,random_state=30),
                                     make_pipeline(
          PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
          SelectFwe(score_func=f_classif, alpha=0.015),
          LogisticRegression(C=20.0, dual=False, penalty="l1")
        ))
      self.model[28] = make_pipeline(LogTransformer(num=0.4),
                                     FactorAnalysis(n_components=12,random_state=30),
                                     make_pipeline(
          PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
          Normalizer(norm="max"),
          MaxAbsScaler(),
          LogisticRegression(C=20.0, dual=False, penalty="l1")
        ))

      self.Ext = ExtraTreesClassifier(bootstrap=False, 
                                      criterion="gini", 
                                      max_features=0.45, 
                                      min_samples_leaf=1, 
                                      min_samples_split=4, 
                                      n_estimators=100)
      self.gradientBoost = GradientBoostingClassifier(learning_rate=0.001, 
                                                      max_depth=5, 
                                                      max_features=0.7500000000000001, 
                                                      min_samples_leaf=18, 
                                                      min_samples_split=15, 
                                                      n_estimators=100, 
                                                      subsample=0.3)

      self.logit = make_pipeline(
        make_union(
          FunctionTransformer(copy),
          FunctionTransformer(copy)
        ),
        LogisticRegression(C=9, class_weight="balanced", fit_intercept=True, max_iter=500, multi_class="multinomial", penalty="l2", solver="sag")
      )
      self.Svc = SVC(C=10, gamma=0.001, kernel="rbf",probability=True)
    
      self.new_model = make_pipeline(
        make_union(
          make_pipeline(
            RobustScaler(),
            PCA(n_components=10,random_state=5)
          ),

          ModelClassTransformer(model=self.model[27])
        ),
        make_pipeline(
          FastICA(tol=0.7000000000000001),
          Normalizer(norm="l2"),
          ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.45, min_samples_leaf=9, min_samples_split=6, n_estimators=100)
        ))
      self.clfstack = make_pipeline(
        make_union(
          make_pipeline(StandardScaler(),
                        PCA(n_components=20,random_state=7)
                       ),
          ModelClassTransformer(self.model[2]),
          ModelClassTransformer(self.model[8]),
          ModelClassTransformer(self.new_model),

        ),
        DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_leaf=8, min_samples_split=12, random_state=10)
      )
      self.pca = make_pipeline(make_pipeline(StandardScaler(),PCA(n_components=15,random_state=41)),
                               make_pipeline(
          SelectPercentile(score_func=f_classif, percentile=72),
          ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.8, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
        ))

      self.clf1 = make_pipeline(
        make_union(
          make_pipeline(
            LogTransformer(num=2), 
            FactorAnalysis(n_components=10, 
                           tol=0.01, 
                           copy=True, 
                           max_iter=3000, 
                           noise_variance_init=None, 
                           svd_method='randomized', 
                           iterated_power=3, 
                           random_state=45)),
          make_pipeline(
            StandardScaler(),
            KernelPCA(n_components=11,kernel="rbf",random_state=7)
          ),
          ModelClassTransformer(self.model[2]),
          ModelClassTransformer(self.model[4]),
          ModelClassTransformer(self.model[7]),
          ModelClassTransformer(self.model[9]),
          ModelClassTransformer(self.model[11]),
          ModelClassTransformer(self.model[12]),
          ModelClassTransformer(self.model[16]),
          ModelClassTransformer(self.model[18]),
          ModelClassTransformer(self.model[20]),
          ModelClassTransformer(self.model[25]),
          ModelClassTransformer(self.model[27]),
          ModelClassTransformer(self.model[28]),
          ModelClassTransformer(self.pca)
        ),
        self.gradientBoost
      )
      self.stack4_bis = make_union(
        FactorAnalysis(n_components=10, 
                       tol=0.01, 
                       copy=True, 
                       max_iter=3000, 
                       noise_variance_init=None, 
                       svd_method='randomized', 
                       iterated_power=3, 
                       random_state=92),
        Blending(self.model[0]),
        Blending(self.model[3]),
        Blending(self.model[6]),
        Blending(self.model[11]),
        Blending(self.model[20]),
        Blending(self.model[19]),
        Blending(self.model[7])
      )
      self.clf2 = make_pipeline(self.stack4_bis,
                               self.Ext)
      
      self.clf = VotingClassifier(estimators=[('clf2', self.clf2), 
                                              ('model[11]', self.model[11]),
                                              ('model[6]', self.model[6]),
                                              ('mlp', self.model[20])
                                             ], 
                                  weights=[2,1,1,2],
                                  voting='soft')




    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)