import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.model_selection import GridSearchCV
import pickle
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn import svm

import OU

save_dir = './data/'
info = np.load(save_dir + "/info.npy")
splits = []
multi_cv_df = pd.DataFrame()
multi_cv_labels = pd.Series()

for i in range(len(info)):
    train = info[i]['train']['df_scale'].copy()
    train_labels = info[i]['train']['labels'].copy()
    
    test = info[i]['test']['df_scale'].copy()
    test_labels = info[i]['test']['labels'].copy()
    
    train_len = train.shape[0]
    test_len = test.shape[0]
    
    # Append rows to dataframe
    multi_cv_df = multi_cv_df.append(train, ignore_index=True)
    multi_cv_labels = multi_cv_labels.append(train_labels, ignore_index=True)
    
    # Append labels to a dataframe
    multi_cv_df = multi_cv_df.append(test, ignore_index=True)
    multi_cv_labels = multi_cv_labels.append(test_labels, ignore_index=True)
    
    # Append the indices of the folds to a list
    splits.append((multi_cv_df.iloc[-train_len-test_len:-test_len].index, multi_cv_df.iloc[-test_len:].index))
    
    # Quality Assurance
    assert(np.array_equal(multi_cv_df.loc[splits[i][0]].values, train.values))
    assert(np.array_equal(multi_cv_labels.loc[splits[i][0]].values, train_labels.values))
    assert(np.array_equal(multi_cv_df.loc[splits[i][1]], test.values))
    assert(np.array_equal(multi_cv_labels.loc[splits[i][1]], test_labels))
    
splits = np.array(splits)

np.save(save_dir + 'splits.npy', splits)

# Save off data
multi_cv_df.to_csv(save_dir + 'df.csv')
multi_cv_labels.to_csv(save_dir + 'labels.csv')

params = {'C': 100, 'cache_size': 2000, 'class_weight': {0: 0.5, 1: 0.5}, 'gamma': 1, 'kernel': 'rbf'}


def custom_kernel(X, Y, primary_kernel, lambda_param, alpha_s, y_s, b, kernel_params=None):
    """
    Custom kernel function based on the provided formula.
    X, Y: Input data matrices.
    primary_kernel: The primary kernel function (e.g., RBF, linear).
    lambda_param: The lambda parameter for D(x).
    alpha_s, y_s, b: Parameters from the support vectors, labels, and bias term.
    kernel_params: Additional parameters for the primary kernel function.
    """
    if primary_kernel == 'rbf':
        K = rbf_kernel(X, Y, **kernel_params)
    elif primary_kernel == 'linear':
        K = linear_kernel(X, Y, **kernel_params)
    else:
        raise ValueError("Unsupported primary kernel")

    D_X = np.exp(-lambda_param * np.square(np.sum(alpha_s * y_s * K + b, axis=1)))
    D_Y = np.exp(-lambda_param * np.square(np.sum(alpha_s * y_s * K.T + b, axis=0)))
    return np.outer(D_X, D_Y) * K


def train_primary_svm(X_train, y_train, kernel='rbf', **kernel_params):
    """
    Train the primary SVM model.
    X_train, y_train: Training data and labels.
    kernel: Kernel type for the primary SVM.
    """
    svm_primary = SVC(kernel=kernel, **kernel_params)
    svm_primary.fit(X_train, y_train)
    return svm_primary


def train_secondary_svm(X_train, y_train, primary_svm, lambda_param, kernel_params=None):
    """
    Train the secondary SVM model with the custom kernel.
    X_train, y_train: Training data and labels.
    primary_svm: The trained primary SVM model.
    lambda_param: The lambda parameter for the custom kernel.
    """
    alpha_s = primary_svm.dual_coef_[0]
    y_s = primary_svm.support_vectors_
    b = primary_svm.intercept_

    svm_secondary = SVC(kernel=lambda X, Y: custom_kernel(X, Y, primary_svm.kernel, lambda_param, alpha_s, y_s, b, kernel_params))
    svm_secondary.fit(X_train, y_train)
    return svm_secondary



ddef evaluate_custom_svm(multi_cv_df, multi_cv_labels, lambda_param, splits, save_dir):
    """
    Evaluate the custom two-step SVM alongside standard SVMs.
    multi_cv_df, multi_cv_labels: Multi-fold training data and labels.
    lambda_param: Lambda parameter for the custom kernel.
    splits: List of (train_index, test_index) tuples for cross-validation.
    save_dir: Directory to save the results.
    """
 
    primary_svm = train_primary_svm(multi_cv_df, multi_cv_labels, kernel='rbf')  # 确保使用RBF核


    custom_svm_kernel = lambda X, Y: custom_kernel(X, Y, 'rbf', lambda_param, 
                                                   primary_svm.dual_coef_[0], 
                                                   primary_svm.support_vectors_, 
                                                   primary_svm.intercept_)

    standard_svm_gridcv = GridSearchCV(SVC(), params, verbose=1, cv=list(splits), n_jobs=-1, 
                                       scoring=['precision'], refit=False)
    standard_svm_gridcv.fit(multi_cv_df, multi_cv_labels)


    with open(save_dir + 'standard_svm_gridsearch_results.pkl', 'wb') as f:
        pickle.dump(standard_svm_gridcv, f)


    custom_svm_gridcv = GridSearchCV(SVC(kernel=custom_svm_kernel), params, verbose=1, cv=list(splits), n_jobs=-1, 
                                     scoring=['precision'], refit=False)
    custom_svm_gridcv.fit(multi_cv_df, multi_cv_labels)


    with open(save_dir + 'custom_svm_gridsearch_results.pkl', 'wb') as f:
        pickle.dump(custom_svm_gridcv, f)


evaluate_custom_svm(multi_cv_df, multi_cv_labels, lambda_param, splits, save_dir)
