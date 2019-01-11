#!/usr/bin/env python

"""
OU class designed to handle the Pandas DataFrame transformation of a OHLC stock ticker dataset according 
the its parameters modelled by the Orstein-Uhlenbeck stochastic process. 
"""

__author__ = "Alex Dai"
__copyright__ = "Patent Pending"
__version__ = "0.1"
__email__ = "alexdai186@gmail.com"
__status__ = "Experimentation"


import pandas as pd
import numpy as np
import scipy.stats
import sklearn


class OU(object):
    """
    Class to handle the train/test splits of the spread of 2 datasets for cross-validation. 
    
    Can handle sliding window cross-validation or expanding window-cross validation
    
    Used to transform the data using the Ornstein Uhlenbeck process to model the residual term.
    
    Fits a Beta on the train split according to the pairs trading spread model to find the residuals 
    of the test set. 
    """
    def __init__(self, df1, df2, model_size=None, eval_size=None):
        """
        Datasets must have equal dimensions.
        
        :df1:           First dataset used for cross-validation. 
        :df2:           Second dataset used for cross-validation.
        """
        self.df1 = df1
        self.df2 = df2
        self.final_df=None
        self.m_size = model_size
        self.e_size = eval_size
        self.fts = []
        self.split_idx = []
        self.splits = []
        
        assert(df1.shape == df2.shape)

    
    def split_expand(self, n_splits=5):
        """
        Finds split indices for expanding window cross-validation.
        
        Implementation taken from:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

        Example split_expand(n_splits=5)

        TRAIN: [0] TEST: [1]
        TRAIN: [0 1] TEST: [2]
        TRAIN: [0 1 2] TEST: [3]
        TRAIN: [0 1 2 3] TEST: [4]
        TRAIN: [0 1 2 3 4] TEST: [5]
        ['CLOSE']
        :num_splits:    How many evaluation periods we want for cross-validation. 
                        Only relevant for expanding window cross-validation. 
                        Defaults to 5
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        self.split_idx = list(tscv.split(self.df1))
        print("Expanding Split Successful.")
    
    
    def split_slide(self, m_size=30000, e_size=10000):
        """
        Finds split indices for sliding window cross-validation.
        
        Example split_sliding(m_size=30000, e_size=10000)
        
        TRAIN: [0:30000] TEST: [30000:40000]
        TRAIN: [10000:40000] TEST: [40000:50000]
        TRAIN: [20000:50000] TEST: [50000:60000]
        TRAIN: [30000:60000] TEST: [60000:70000]
        TRAIN: [40000:70000] TEST: [70000:80000]
        
        :model_size:    How large of training model we want to use for sliding window cross-validation. 
                        This parameter will be None if expanding window flag is set. 
        :eval_size:     How large of a testing model we want our sliding window cross-validation fit will 
                        be evaluated on. 
                        This parameter will be none if expanding window flag is set. 
        """
        splits = []
        end_ind = m_size
        cur_ind = 0
        
        assert(m_size < self.df1.shape[0])
        
        while end_ind < self.df1.shape[0]:
            # Finds train indices 
            train_ind = np.array(np.arange(cur_ind, end_ind))
            
            # If test indices for last test split less than e_size, then just use the rest.
            if end_ind+e_size<self.df1.shape[0]:
                test_ind = np.array(np.arange(end_ind, end_ind+e_size))  
            else:
                test_ind = np.array(np.arange(end_ind, self.df1.shape[0]))
                
            splits.append((train_ind, test_ind))
            end_ind += e_size
            cur_ind += e_size
        
        self.split_idx = splits
        print("Sliding Window Split Successful.")
    
    
    def fit_feature(self, s1, s2, feature):
        """
        This method takes in the features of two different stocks, calculates the residuals, 
        runs lag 1 auto-regression, then estimates parameters for the original OU process equation, which 
        we will then use to normalize the features into a T-score

        :s1:         Slice of first ticker feature vec.
        :s2:         Slice of second ticker feature vec.
        :feature:    Feature to model OU Process on. 
        :window:     Size ma_window used so that we know where the NaNs end. 
        
        :ret: (fitted feature df, transformed test df)
        """
                
        s1 = s1[feature]
        s2 = s2[feature]
        
        # Estimate linear relationship between p1 and p2 using linear regression
        beta, dx, _, _, _ = scipy.stats.linregress(s2, s1)
        # Retrieve the residuals (dx_t) of our estimate

        
        residuals = s1 - (s2*beta)

        # Integrate dx_t to find x_t
        x_t = np.cumsum(residuals)
        lag_price = x_t.shift(1)
        
        # Perform lag-1 auto regression on the x_t and the lag
        b, a, _, _, _ = scipy.stats.linregress(lag_price.iloc[1:], x_t.iloc[1:])

        # Calculate parameters to create a t-score
        # theta = -1/math.log(b)
        mu = a/(1-b)
        sigma = np.sqrt(np.var(x_t))

        t_score = (x_t - mu)/sigma
        t_score.name = feature
        # Return absolute value of t_score because we only care about the spread
        # t_score = np.abs(t_score)

        # Return transformed vector, residuals, beta, and the index of the dataframe we fit on.
        return {'tscore_fit_'+feature: t_score, 'residuals_fit_'+feature: residuals, 
                'beta_fit_'+feature: beta, 'dx_fit_'+feature: dx, 
                'mu_fit_'+feature: mu, 'sigma_fit_'+feature: sigma, 
                'fit_index_'+feature: np.array(s1.index)}
    
    
    def transform(self, t1, t2, feature, fit_dict):
        """
        Transforms the target feature vector slices using the OU model parameters obtained
        in the fit() method. 
        
        :t1:         Slice of first ticker feature vec.
        :t2:         Slice of second ticker feature vec.
        :fit_dict:   Dictionary of parameter values. 
        """
        beta = fit_dict['beta_fit_'+feature]
        dx = fit_dict['dx_fit_'+feature]
        mu = fit_dict['mu_fit_'+feature]
        sigma = fit_dict['sigma_fit_'+feature]
        
        s1 = t1[feature]
        s2 = t2[feature]
        
        residuals = s1 - (s2*beta)

         # Integrate dx_t to find x_t
        x_t = np.cumsum(residuals)

        # Calculate parameters to create a t-score
        t_score = (x_t - mu)/sigma
        
        # Return absolute value of t_score because we only care about the spread
        # t_score = np.abs(t_score)
        t_score.name = feature
        
        # Return transformed vector, residuals, beta, and the index of the dataframe we fit on.
        return {'tscore_transform_'+feature: t_score, 'residuals_transform_'+feature: residuals, 
                'transform_index_': np.array(t1.index)}
    
    
    def fit_transform(self, d1, d2, t1, t2, ou_features, other_features=None):
        """
        This method takes in the features of two different stocks, calculates the residuals, 
        runs lag 1 auto-regression, then estimates parameters for the original OU process equation, which 
        we will then use to normalize the features into a T-score

        :d1:             First ticker dataframe for fitting.
        :d2:             Second ticker dataframe for fitting.
        :t1:             First ticker dataframe for transforming.
        :t2:             Second ticker dataframe for transforming. 
        :ou_features:    List of features meant for OU parameterization.
        :all_features:   List of features meant to be retained in overall df. 
        
        :ret: (fitted train dataframe, transformed test df)
        """
        fit_dicts = {}
        t_dicts = {}
        
        for feature in ou_features:
            fit_dict = self.fit_feature(d1, d2, feature)
            fit_dicts.update(fit_dict)
            
            t_dict = self.transform(t1, t2, feature, fit_dict)
            t_dicts.update(t_dict)
            
        train = pd.DataFrame([fit_dicts[f] for f in fit_dicts.keys() if 'tscore' in f]).T
        test = pd.DataFrame([t_dicts[t] for t in t_dicts.keys() if 'tscore' in t]).T
        
        if other_features:
            for feat in other_features:
                train[feat+'1'] = d1[feat]
                train[feat+'2'] = d2[feat]
                test[feat+'1'] = t1[feat]
                test[feat+'2'] = t2[feat]

        return {'train': {'df': train, **fit_dicts}, 'test': {'df': test, **t_dicts}}
        
        
    def get_splits(self, ou_features, other_features=None, label_func=None, scale=False):
        """
        Returns final list of all fit and transformed dfs and corresponding fit dictionaries. 
        
        :ou_features:       determines which features you want to transform according to OU model. 
        :other_features:    determines other features you want to keep for your model that aren't 
                            transformed according to OU model. 
        :label_func:            Labelling function we want to apply to our dataset. 
        """
        assert(self.split_idx)
        
        fts = []
        # Fit-Transform the train and test datasets for each of the splits. 
        for train, test in self.split_idx:
            df_train1 = self.df1.loc[train]
            df_train2 = self.df2.loc[train]
            df_test1 = self.df1.loc[test]
            df_test2 = self.df2.loc[test]
            ft = self.fit_transform(df_train1, df_train2, df_test1, df_test2, ou_features, other_features)
            ft['train']['index'] = train
            ft['test']['index'] = test

            # Create Labels
            if label_func:
                train_labels = label_func(ft['train']['residuals_fit_price'])
                test_labels = label_func(ft['test']['residuals_transform_price'])
                ft['train']['labels'] = train_labels
                ft['test']['labels'] = test_labels
                
            # Perform Feature Scaling
            if scale:
                min_max_scaler = sklearn.preprocessing.MinMaxScaler()
                x_scaled = min_max_scaler.fit_transform(ft['train']['df'])
                y_scaled = min_max_scaler.transform(ft['test']['df'])
                df_scaledx = pd.DataFrame(x_scaled)
                df_scaledy = pd.DataFrame(y_scaled)
                ft['train']['df_scale'] = df_scaledx
                ft['test']['df_scale'] = df_scaledy
                
            fts.append(ft) 
        self.fts = fts
        return fts
