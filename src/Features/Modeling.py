# Import Packages
import numpy as np
import pandas as pd
import pickle
import os
from itertools import compress
import random
import datetime
import paramiko
from dotenv import load_dotenv
import copy

# Import sklearn
from scipy.stats.stats import pearsonr, spearmanr, ttest_ind
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression

# Import Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow import metrics
from tensorflow.keras import layers, metrics, losses, optimizers
import keras_tuner as kt
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow_probability import stats

# Import Functions
from Features.Algorithms import *

# Functions
def count_ps(X, y):
    count = 0
    for col in X:
        r,p = pearsonr(X[col], y)
        if p < 0.05:
            count = count + 1
    return count

def two_binner(y = None, thresh = None):
    if y >= thresh:
        return 1
    else:
        return 0

class DataSet():

    # CLASS METHODS
    def __init__(self, csv):
        self.csv  = csv

    def load_dataframe(self, sample_size = None, num_noise_levels = 10, parent_path = None):
        """
        Loads instance variables for a dataSheet instance.

        Parameters:
        csv (CSV): Descriptor file to load.
        sample_size (int): Number of entries to sample from the dataset. (Default = None)
        num_noise_levels (int): Number of noise levels to generate. (Default = 10)
        parent_path (string): Custom folder to save results in, if desired. (Default = None)

        :return:
        """

        # Load CSV into df variable
        self.df = pd.read_csv(filepath_or_buffer=self.csv, index_col=0)

        # Set name variable
        self.name = self.csv.split('\\')[-1].split('.')[0].split('_')[0]

        # Set sample size class variable to the number of rows if None
        if sample_size is None:
            sample_size = self.df.shape[0]

        # Exit function if sample size is too big
        if sample_size > self.df.shape[0]:
            message = 'The sample_size must be smaller than the size of the dataset, which is: {}'.format(self.df.shape[0])
            return print(message)

        # Sample the dataframe
        if sample_size:
            self.df = self.df.sample(sample_size)
        
        # Set sample_size variable
        self.sample_size = sample_size

        # Drop Infs
        self.df = drop_infs(self.df)

        # Select y variables
        self.y_true = self.df.iloc[:, -1]

        # Sample noise and add to variable
        self.y_dict = sample_noise(y = self.y_true, num_levels = num_noise_levels)

        # Set optional parent_path variables
        self.parent_path = parent_path

        # Set X and other dataset variables
        self.X = self.df.iloc[:, :-1]
        self.num_features = len(self.X.columns)
        self.features = self.X.columns

        # Set empty splitting and train test split variables
        self.k_folds = None
        self.splitting = None
        self.training_sets = {'X_train': [], 'y_train': []}
        self.testing_sets = {'X_test': [], 'y_test': []}

        # Tups variable is a list, each member corresponds to a set of algorithms optimized
        # on the respective training set, in the same order
        self.tups = []
        
        # Set preprocessing variables to false
        self.var_filter_ = {'filtered': False, 'Value': None}
        self.random_features_ = {'filtered': False, 'Value': None}
        self.drop_corr_ = {'filtered': False, 'Value': None}
        self.scaled_ = False
        self.select_k_best_ = {'filtered': False, 'Value': None}
        self.optimized_on_ = 'Not Optimized'

        return None
    
    def k_fold_split_train_test(self, n_folds, split_method = 'Random'):
        """
        Forms k-folds from a dataset and assigns class variables self.training_sets and self.testing_sets to
        the respective training and test sets within each fold.

        """

        # Set class variables
        self.k_folds = n_folds
        self.splitting = split_method

        # Add Clause for KFold versus StratifiedKFold
        if split_method == 'Stratified':
            skf = StratifiedKFold(n_splits=n_folds)
            skf.get_n_splits(self.X, self.y_true)

        elif split_method == 'Random':
            skf = KFold(n_splits= n_folds)
            skf.get_n_splits(self.X, self.y_true)

        # Loop through splits
        for train_index, test_index in skf.split(self.X, self.y_true):
            # Get training and test sets from indices
            X_train = self.X.iloc[train_index, :]
            y_train = self.y_true.iloc[train_index]
            X_test = self.X.iloc[test_index, :]
            y_test = self.y_true.iloc[test_index]

            # Set class variables
            self.training_sets['X_train'].append(X_train)
            self.training_sets['y_train'].append(y_train)
            self.testing_sets['X_test'].append(X_test)
            self.testing_sets['y_test'].append(y_test)

        return None

    def split_train_test(self, percentage):
        """
        Forms a single training and test set from a dataset.
        """

        # Raise error if kfold splitting already set
        if self.k_folds or self.splitting:
            return print("Cannot use train test split and kfold splitting simultaneously.")

        # split dataset into train and test sets
        training_df = self.X.sample(frac=percentage)
        test_df = self.X.loc[~self.X.index.isin(training_df.index)]
        self.training_sets['X_train'].append(training_df)
        self.training_sets['y_train'].append(self.y_true.iloc[training_df.index])
        self.testing_sets['X_test'].append(test_df)
        self.testing_sets['y_test'].append(self.y_true.iloc[test_df.index])

        # Set other conflicting variables
        self.k_folds = None
        self.splitting = None
    
        return None

    def make_meta_func(self):
        """
        Takes all the attributes of a dataset object and puts them into an external meta dictionary.

        Parameters:
            dataset (dataset object): DataSet class object.

        Returns:
            meta_dict (dict): Dictionary containing all the dataset attributes.
        """
        meta_dict =  copy.deepcopy(self.__dict__)
        for key in ['csv', 'df', 'X', 'y_true', 'y_dict']:
            del meta_dict[key]
        
        # # Set the training sets to only be indices to save space
        meta_dict['training_sets']['X_train'] = [x.index for x in meta_dict['training_sets']['X_train']]
        meta_dict['testing_sets']['X_test'] = [x.index for x in meta_dict['testing_sets']['X_test']]

        # Convert each estimator within self.tups into a dictionary of parameters to save space
        new_meta = {}
        new_meta['tups'] = []
        for alg_tup in meta_dict['tups']:
            clf = alg_tup[0]
            rgr = alg_tup[1]
            alg_name = alg_tup[2]
            params_tuple = (clf.get_params(deep=True), rgr.get_params(deep=True), alg_name)
            new_meta['tups'].append(params_tuple)
        meta_dict['tups'] = new_meta['tups']

        return meta_dict

    def generate_data(self, test_set = 'True', optimize_on = ['Continuous', 'Categorical']):
        """
        Loops through each y column stored in y_dict (generated from sampleNoise()), generates 9
        binary splits on percentiles 10 through 90, and trains a classifier/regressor pair on each fold. Takes the average BA score for each fold
        for classifier, turns the continuous regression predictions into classes, and takes average BA score
        from those. Saves the data in a PKL file.

        Parameters:
            tups (list of tuples): List of tuples containing classifier, regressor, name for several algorithms.
            test_set (str): Can be 'True' or 'Noise'. Determines whether the test set for modeling will have noise or not. (Default = 'True')
            use_ssh (bool): If True the files will be saved over SSH, if false they will be saved locally. (Default = False)
        Returns:

        """
        # Set class variables from function inputs
        self.test_set = test_set

        # Set function variables from dataset class variables
        y_dict = self.y_dict
        X = self.X
        y_true = self.y_true
        dataset_name = self.name
        sample_size = self.sample_size
        num_features = self.num_features
        k_folds = self.k_folds

        # Loop through noise levels
        for lvl_dict in y_dict.keys():
            y = y_dict[lvl_dict]['y']
            sigma = y_dict[lvl_dict]['sigma']
            noise_level = str(lvl_dict)

            # Define thresholds for looping
            threshes = [np.percentile(y, x) for x in [10, 20, 30, 40, 50, 60, 70, 80, 90]]
            threshtup = list(zip(threshes, [10, 20, 30, 40, 50, 60, 70, 80, 90]))

            # Loop through each algorithm classifier/regressor pair
            for alg_tup in self.tups:
                
                # Set clf, rgr, and alg_name
                clf = alg_tup[0]
                rgr = alg_tup[1]
                alg_name = alg_tup[2]

                # Define meta dictionary from dataset attributes
                meta = self.make_meta_func()

                # Fill in meta dictionary from loop specific variables
                meta['Thresholds'] = threshtup
                meta['Algorithm'] = alg_name
                meta['Noise Level'] = noise_level
                meta['Sigma'] = sigma
                meta['Estimator'] = alg_tup
                meta['training_set_params'] = []

                # Create list which contains scores for each training set
                alg_score_list = []

                # Loop through training sets
                for i, training_set in enumerate(self.training_sets['X_train']):
                    X_train = self.training_sets['X_train'][i]
                    y_train = self.training_sets['y_train'][i]
                    X_test = self.testing_sets['X_test'][i]
                    y_test = self.testing_sets['y_test'][i]

                    print(f'TRAINING FOLD {i} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

                    # Optimize algorithm if needed
                    if optimize_on:
                        self.optimize_alg(tup = alg_tup, X = X_train, y = y_train, meta_dict = meta, optimize_on = optimize_on)
                    else:
                        meta['training_set_params'].append(rgr.get_params(deep=True))

                    # Create list of score dicts which contains scores for all thresholds
                    fold_score_dict_list = []

                    # Loop through thresholds
                    for thresh, perc in threshtup:

                        # Print statement
                        print(f'{perc} Percentile ###################################')

                        # Get score for single fold and single threshold
                        thresh_score_dict = get_clf_rgr_scores_test_set(thresh=thresh, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, clf = clf, rgr = rgr)
                        fold_score_dict_list.append(thresh_score_dict)

                        # Print statement
                        print(thresh_score_dict)
                        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    
                    # Append the list of threshold scores for entire fold to alg score list
                    alg_score_list.append(fold_score_dict_list)

                # Save accumulated score dictionary
                print('Saving ###################################################################')
                save_pkl(score_dict = alg_score_list, meta = meta, pkl_path = self.parent_path)

        return None

#---------------------
# PREPROCESSING
#---------------------
    def scale_x(self):
        """

        Uses StandardScaler() to scale X data of the dataset.
        """
        # Give error if train test split already performed
        if self.training_sets['X_train']:
            return print("Must scale X before train test split.")

        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(self.X)
        scaled_df = pd.DataFrame(scaled_X, index = self.X.index, columns = self.X.columns)
        self.X = scaled_df
        self.scaled_ = True

        return None

    def drop_low_variance_features(self, threshold = 0):
        """

        Drops features with variance below a certain threshold
        """
        # Give error if train test split already performed
        if self.training_sets['X_train']:
            return print("Must drop low variance features before train test split")

        # selector = VarianceThreshold(threshold=threshold)
        # selector.fit_transform(self.X)

        # Get variance and assign to dataframe where feature variables are rows
        variance = self.X.var()
        df_var = pd.DataFrame(data = {'variance': variance}, index = self.X.columns)

        # Drop the low variance rows
        df_low_v_dropped = df_var[df_var['variance'] > threshold]

        # Filter the dataset's X dataframe by the selected feature variables
        self.X = self.X[df_low_v_dropped.index]

        # Reassign dataset variables
        self.num_features = len(self.X.columns)
        self.features = self.X.columns
        self.var_filter_ = {'filtered': True, 'Value': threshold}

        return None
    
    def drop_correlated_features(self, threshold = 0.95):
        """

        Drops correlated feature variables above the supplied threshold.
        """
        # Give error if train test split already performed
        if self.training_sets['X_train']:
            return print("Must drop correlated features before train test split.")

        # Create correlation matrix
        corr_matrix = self.X.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        # Drop features
        self.X.drop(to_drop, axis=1, inplace=True)

        # Reassign dataset variables
        self.num_features = len(self.X.columns)
        self.features = self.X.columns
        self.drop_corr_ = {'filtered': True, 'Value': threshold}

        return None
    
    def select_k_best_features(self, k_features):
        """

        Selects the best k features using f score.
        """
        # Select the k best features
        selector = SelectKBest(f_regression, k = k_features)
        X_new = selector.fit_transform(self.X, self.y_true)
        X_col_nums = selector.get_support(indices=True)

        # Update dataset variables
        self.X = self.X.iloc[:, X_col_nums]
        self.num_features = len(self.X.columns)
        self.features = self.X.columns
        self.select_k_best_ = {'filtered': True, 'Value': k_features}

        return None
    
    def compare_feature_statistics(self):
        """

        Use statistics function of choice to get scores and p-values for each feature in a dataset.
        """

        # Compute continuous stats
        f_stats, p_vals = f_regression(self.X,self.y_true)
        m_i = mutual_info_regression(self.X, self.y_true)
        sp_corrs = []
        sp_ps = []
        for col in self.X.columns:
            sp_corr, sp_p = spearmanr(self.X[col], self.y_true)
            sp_corrs.append(sp_corr)
            sp_ps.append(sp_p)

        # Vectorize binning function
        twobin_v = np.vectorize(two_binner)

        # Define y_class and compute class stats
        thresh = np.median(self.y_true)
        y_class = twobin_v(self.y_true, thresh=thresh)
        f_stats_class, p_vals_class = f_classif(self.X,y_class)
        m_i_class = mutual_info_classif(self.X, y_class)
        sp_corrs_class = []
        sp_ps_class = []
        for col in self.X.columns:
            sp_corr_class, sp_p_class = spearmanr(self.X[col], y_class)
            sp_corrs_class.append(sp_corr_class)
            sp_ps_class.append(sp_p_class)

        # Create dataframe
        df_stats = pd.DataFrame(data = {
            'f_stats': f_stats,
            'p_vals': p_vals,
            'm_i': m_i,
            'sp_corrs': sp_corrs,
            'sp_ps': sp_ps,
            'f_stats_class': f_stats_class,
            'p_vals_class': p_vals_class,
            'm_i_class': m_i_class,
            'sp_corrs_class': sp_corrs_class,
            'sp_ps_class': sp_ps_class
            },
            index = self.X.columns)

        return df_stats
#-------------------------
# ALGORITHM STEPS
#-------------------------
    def make_algs(self):
        """
        Sets non optimized algorithm tuples to class variable self.tups.
        """
        # Instantiate Classifiers
        knn = KNeighborsClassifier(weights='distance')
        dt = DecisionTreeClassifier(max_features= 'auto', criterion= 'gini')
        svc = SVC(probability=True)
        rf = RandomForestClassifier(criterion= 'gini')
        clf_list = [knn, dt, svc, rf]

        # Instantiate Regressors
        knnr = KNeighborsRegressor(weights='distance')
        dtr = DecisionTreeRegressor(max_features= 'auto', criterion= 'mse')
        svr = SVR()
        rfr = RandomForestRegressor(criterion= 'mse')
        rgr_list = [knnr, dtr, svr, rfr]

        # Reinstantiate as new tuple and add to list
        names = ['KNN', 'DT', 'SVM', 'RF']
        tups = zip(clf_list, rgr_list, names)
        tup_list = list(tups)
        self.tups.extend(tup_list)

        return None
    
    # DEPRECATED
    def make_opt_algs(self, X, y, optimize_on = ['Continuous', 'Categorical']):
        """

        Optimizes algorithms via GridSearchCV and returns a tuple of optimized clfs and rgrs.

        Inputs:
        optimize_on (String): 'Continuous' will optimize on continuous dataset, and 'Categorical' will optimize on categorical dataset.
        """

        # Instantiate Classifiers
        knn = KNeighborsClassifier(weights='distance')
        dt = DecisionTreeClassifier(max_features= 'auto', criterion= 'gini')
        svc = SVC(probability=True)
        rf = RandomForestClassifier(criterion= 'gini')
        clf_list = [knn, dt, svc, rf]

        # Instantiate Regressors
        knnr = KNeighborsRegressor(weights='distance')
        dtr = DecisionTreeRegressor(max_features= 'auto', criterion= 'mse')
        svr = SVR()
        rfr = RandomForestRegressor(criterion= 'mse')
        rgr_list = [knnr, dtr, svr, rfr]

        # Define rgr param grids
        dt_dict = {
            'max_depth': [50, 100, 200, 500],
            'min_samples_split': [2, 5, 10, 20, 40],
            'min_samples_leaf': [1, 5, 10, 20]
            }

        knn_dict = {
            'n_neighbors': np.arange(2, 22, 1)
            }

        svm_dict = {
            'kernel': ['sigmoid', 'rbf'],
            'C': [0.001, 0.01, 0.1, 1, 10]
            }

        rf_dict = {
            'n_estimators': [10, 25, 50, 100, 150, 200],
            'max_depth': [50, 100, 200, 500],
            'min_samples_split': [2, 5, 10, 20, 40],
            'min_samples_leaf': [1, 5, 10, 20]
            }

        if optimize_on == 'Continuous':
            
            # Make rgr GridSearch
            knn_search = GridSearchCV(estimator = knnr, param_grid = knn_dict, n_jobs = -1, verbose= 10)
            dt_search = GridSearchCV(estimator = dtr, param_grid = dt_dict, n_jobs = -1, verbose= 10)
            svm_search = GridSearchCV(estimator = svr, param_grid = svm_dict, n_jobs = -1, verbose= 10)
            rf_search = GridSearchCV(estimator = rfr, param_grid = rf_dict, n_jobs = -1, verbose= 10)
            searches = [knn_search, dt_search, svm_search, rf_search]

            # Set dataset variable
            self.optimized_on_ = 'Continuous'

        elif optimize_on == 'Categorical':

            # Set scoring function as multiple metrics
            scoring = ['balanced_accuracy', 'f1', 'roc_auc', 'neg_brier_score']

            # Make clf GridSearch
            knn_search = GridSearchCV(estimator = knn, param_grid = knn_dict, n_jobs = -1, verbose= 10, scoring= scoring, refit = 'f1')
            dt_search = GridSearchCV(estimator = dt, param_grid = dt_dict, n_jobs = -1, verbose= 10, scoring= scoring, refit = 'f1')
            svm_search = GridSearchCV(estimator = svc, param_grid = svm_dict, n_jobs = -1, verbose= 10, scoring= scoring, refit = 'f1')
            rf_search = GridSearchCV(estimator = rf, param_grid = rf_dict, n_jobs = -1, verbose= 10, scoring= scoring, refit = 'f1')
            searches = [knn_search, dt_search, svm_search, rf_search]

            # Set y to be categorical
            # Vectorize binning function
            twobin_v = np.vectorize(two_binner)
            thresh = np.median(y)
            y = twobin_v(y, thresh = thresh)

            # Set dataset variable
            self.optimized_on_ = f'Categorical: {thresh}'

        # Make names and zip into tuples
        gridnames = ['KNNOpt', 'DTOpt', 'SVMOpt', 'RFOpt']

        # Fit searches
        for i,search in enumerate(searches):
            
            # Fit the gridsearch
            search.fit(X, y)
            opt_params = search.best_params_

            # Set the parameters of the regressor
            rgr = rgr_list[i]
            rgr.set_params(**opt_params)

            # Set the parameters of the classifier
            clf = clf_list[i]
            clf.set_params(**opt_params)

        # Reinstantiate as new tuple and add to list
        names = ['KNN', 'DT', 'SVM', 'RF']
        tups = list(zip(clf_list, rgr_list, names))

        # Append to class variable
        self.tups.append(tups)

        return None
    
    def optimize_alg(self, tup, X, y, meta_dict, optimize_on = ['Continuous', 'Categorical']):
        """ Optimizes an algorithm. """

        # Assign clf and rgr
        clf = tup[0]
        rgr = tup[1]
        alg_name = tup[2]

        # Check which algorithm to assign optimization dictionary
        if alg_name == 'KNN':
            opt_dict = {
            'n_neighbors': np.arange(2, 22, 1)
            }
        
        elif alg_name == 'DT':
            opt_dict = {
            'max_depth': [50, 100, 200, 500],
            'min_samples_split': [2, 5, 10, 20, 40],
            'min_samples_leaf': [1, 5, 10, 20]
            }
        
        elif alg_name == 'SVM':
            opt_dict = {
            'kernel': ['sigmoid', 'rbf'],
            'C': [0.001, 0.01, 0.1, 1, 10]
            }

        elif alg_name == 'RF':
            opt_dict = {
            'n_estimators': [10, 25, 50, 100, 150, 200],
            'max_depth': [50, 100, 200, 500],
            'min_samples_split': [2, 5, 10, 20, 40],
            'min_samples_leaf': [1, 5, 10, 20]
            }

        if optimize_on == 'Continuous':
            
            # Make rgr GridSearch
            search = GridSearchCV(estimator = rgr, param_grid = opt_dict, n_jobs = -1, verbose= 5)

            # Set dataset variable
            self.optimized_on_ = 'Continuous'

        elif optimize_on == 'Categorical':

            # Set scoring function as multiple metrics
            scoring = ['balanced_accuracy', 'f1', 'roc_auc', 'neg_brier_score']

            # Make clf GridSearch
            search = GridSearchCV(estimator = clf, param_grid = opt_dict, n_jobs = -1, verbose= 5, scoring= scoring, refit = 'f1')

            # Set y to be categorical
            # Vectorize binning function
            twobin_v = np.vectorize(two_binner)
            thresh = np.median(self.y_true)
            y = twobin_v(y, thresh = thresh)

            # Set dataset variable
            self.optimized_on_ = f'Categorical: {thresh}'

            # Set meta dictionary value
            meta_dict['optimized_on'] = self.optimized_on_

        # Fit the gridsearch
        search.fit(X, y)
        opt_params = search.best_params_

        # Set the parameters of the regressor
        rgr.set_params(**opt_params)

        # Set the parameters of the classifier
        clf.set_params(**opt_params)

        # Add the current params to a meta_dict key corresponding to the correct training set
        meta_dict['training_set_params'].append(rgr.get_params(deep=True))

        return None
    
    def make_dnns(self):
        """ Makes DNN rgr/clf pair and adds to tups attribute."""

        # Define dnn_tuner directories
        self.dnn_directories = {}
        self.dnn_directories['clf_directory'] = r'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\DNN_Tuner_tmp\Clf_tmp'
        self.dnn_directories['rgr_directory'] = r'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\DNN_Tuner_tmp\Rgr_tmp'

        # Instantiate Classifier
        dnn_clf = class_dnn_model_builder(hp=None, output_bias = None)

        # Instantiate Regressor
        dnn_rgr = reg_dnn_model_builder(hp=None)

        # Instantiate Name
        dnn_name = 'DNN'

        # Add DNN tup to class attribute
        dnn_tup = (dnn_clf, dnn_rgr, dnn_name)
        self.tups.append(dnn_tup)

        return None

#----------------------------
# NON CLASS METHODS
#----------------------------
def random_x(df, num):
    """
    Selects a random X variable.

    Parameters:
    df (dataframe): Dataframe to select from.
    num (int): Number of features to randomly select.

    Returns:
        feature (Series name)
    """
    # Identify X variable columns
    X = df.iloc[:, :-1]

    # Select a random variable
    columns = list(X.columns)
    feature = random.sample(columns, num)

    return feature

def drop_infs(df=None):
    """
    Drops columns which contain infinite values in a dataframe.

    Parameters:
    df (dataframe): Dataframe to drop infinite values. (Default = 600)

    Returns:
        df (dataframe): Dataframe with infinite values dropped.
    """
    cols = [x for x in df.columns if df[x].dtype == 'object']
    df = df.drop(columns=cols)
    df[df.columns] = df[df.columns].astype(float)
    df = df.fillna(0.0).astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how='any')
    return df

def sample_noise(y = None, base_sigma = None, num_levels = 10, scaling_factor = 0.01):
    """
    Generates 'num_levels' levels of gaussian distributed noise calculated from the integer range of y values in the
    input y variable.

    Parameters:
    y (dataframe series): Continuous y variable for a descriptor dataframe. (Default = 600)
    base_sigma (float): Base value to generate noise levels from. If set to 600, will be the range of endpoint values.
                        (Default = 600)
    num_levels (int): Integer value that gives the number of noise levels to generate. (Default = 10)
    scaling_factor (float): Factor to scale the added noise, typically around 0.01. (Default = 0.01)

    Returns:
    end_dict (dict of series): Dictionary of endpoint columns with added gaussian distributed noise.
    end_dict = {'Noise_x': {'y': pandas series with error added to y values, 'sigma': sigma for added noise}}

    """

    # Get range of y_int values
    y_max = y.max()
    y_min = y.min()

    y_range = y_max-y_min
    if base_sigma is None:
        multiplier = y_range
    else:
        multiplier = base_sigma

    # Create list of noise levels and choose multiplier
    lvls = [*range(0, num_levels, 1)]
    noiselvls = [scaling_factor*multiplier*x for x in lvls]

    # Create a dict of endpoints, each with increasing noise
    end_dict = {}
    for i,lvl in enumerate(noiselvls):

        # Get size of endpoint columns for gaussian number generator
        col_size = y.shape[0]

        # Generate gaussian distributed noise which is the same shape as endpoint column
        err = np.random.normal(0, scale = lvl, size = [col_size])

        # Make unique names endpoint column by level of noise
        col_name = 'Noise_' + str(i)

        # Add new endpoint column to dictionary
        end_dict[col_name] = {}
        end_dict[col_name]['y'] = y.copy(deep = True) + err
        end_dict[col_name]['sigma'] = lvl

    return end_dict

def get_clf_rgr_scores_test_set(thresh, X_train, X_test, y_train, y_test, clf, rgr):
    """
    Takes a meta-dictionary, training and test set. Fits a classifier and gets a classifier score. Fits a regressor, predicts, converts predictions
    to classes based on threshold, and gets score on those classes. All for a single algorithm (clf/rgr pair).

    Parameters:
        meta (dict): Python dictionary containing meta-information. (Default = 600).
        train (Pandas dataframe): Training set.
        test (Pandas dataframe): Test set.

    Returns:
         BA (list): List of meta dictionary, classifier score dictionary, and regressor score dictionary.
    """

    # Vectorize binning function
    twobin_v = np.vectorize(two_binner)

    # Define categorized variables
    y_train_class = pd.DataFrame(twobin_v(y_train, thresh = thresh), index= y_train.index)
    y_train_class_vals = y_train_class.iloc[:].values.ravel()
    y_test_class = pd.DataFrame(twobin_v(y_test, thresh=thresh), index=y_test.index)

    # Fit Clf
    if type(clf) == 'keras.engine.sequential.Sequential':
        num_neg = _
        num_pos = _
        num_total = _
        class_weight = {0: (1/num_neg) * (num_total/2), 1: (1/num_pos) * (num_total/2)}
        clf.fit(X_train, y_train, validation_split=0.2, verbose=0, epochs=200, class_weight=class_weight)

    else:
        clf.fit(X_train, y_train_class_vals)

    # Classifier probabilistic scores
    y_prob_clf = clf.predict_proba(X_test)
    roc_score_clf = roc_auc_score(y_test_class, y_prob_clf[:,1])
    brier_score_clf = brier_score_loss(y_test_class, y_prob_clf[:,1])
    logloss_score_clf = log_loss(y_test_class, y_prob_clf[:,1])

    # Classifier regular scores
    y_pred_clf = clf.predict(X_test)
    BA_score_clf = balanced_accuracy_score(y_test_class, y_pred_clf)
    F1_score_clf = f1_score(y_test_class, y_pred_clf)
    kappa_score_clf = cohen_kappa_score(y_test_class, y_pred_clf)
    pearsphi_score_clf = matthews_corrcoef(y_test_class, y_pred_clf)

    # Fit and predict with rgr and get score
    log = LogisticRegression(max_iter= 1000, solver = 'liblinear')
    rgr.fit(X_train, y_train.values.ravel())
    y_pred_rgr = rgr.predict(X_test)
    y_pred_rgr_class = twobin_v(y_pred_rgr, thresh=thresh)

    # Try except block to handle ValueError for having only a single class in a test set
    logregerror = False
    try:
        log.fit(X_test, y_pred_rgr_class)
        y_prob_rgr = log.predict_proba(X_test)
    except ValueError:
        print('One class in Logistic Regressions Test set')
        logregerror = True
        pass

    # Add if clause for if Logistic Regression passed
    if logregerror:
        roc_score_rgr = np.nan
        brier_score_rgr = np.nan
        logloss_score_rgr = np.nan
    else:
        # Get rgr probabilistic scores
        roc_score_rgr = roc_auc_score(y_test_class, y_prob_rgr[:,1])
        brier_score_rgr = brier_score_loss(y_test_class, y_prob_rgr[:, 1])
        logloss_score_rgr = log_loss(y_test_class, y_prob_rgr[:, 1])

    # Get rgr regular scores
    BA_score_rgr = balanced_accuracy_score(y_test_class, y_pred_rgr_class)
    F1_score_rgr = f1_score(y_test_class, y_pred_rgr_class)
    kappa_score_rgr = cohen_kappa_score(y_test_class, y_pred_rgr_class)
    pearsphi_score_rgr = matthews_corrcoef(y_test_class, y_pred_rgr_class)

    score_dict = {
    'BA': {'Clfs': BA_score_clf, 'Rgrs': BA_score_rgr},
    'F1': {'Clfs': F1_score_clf, 'Rgrs': F1_score_rgr},
    'ROC-AUC': {'Clfs': roc_score_clf, 'Rgrs': roc_score_rgr},
    'Brier': {'Clfs': brier_score_clf, 'Rgrs': brier_score_rgr},
    'Kappa': {'Clfs': kappa_score_clf, 'Rgrs': kappa_score_rgr},
    'Logloss': {'Clfs': logloss_score_clf, 'Rgrs': logloss_score_rgr},
    'Pearsphi': {'Clfs': pearsphi_score_clf, 'Rgrs': pearsphi_score_rgr}
    }

    return score_dict

def accumulate_scores(score_dict_list):
    """
    Takes individual scores from different folds and accumulates into an accumulated score dictionary.
    """

    # Extract individual scores from score list and refactor for a save-able version
    # Create empty dictionary for score reporting
    accumulated_scores = {}
    for key in score_dict_list[0].keys():
        accumulated_scores[key] = {'Clfs': [], 'Rgrs': []}

    # Accumulate the scores from the individual score dictionaries
    for score_dict in score_dict_list:
        for key in score_dict.keys():
            clf_score = score_dict[key]['Clfs']
            accumulated_scores[key]['Clfs'].append(clf_score)
            rgr_score = score_dict[key]['Rgrs']
            accumulated_scores[key]['Rgrs'].append(rgr_score)

    return accumulated_scores

def save_pkl(score_dict = None, meta = None, pkl_path = None):
    """
    Takes a list BA which contains a meta-dictionary and algorithm metrics,
    and saves a PKL file with all the information.

    Parameters:
        score_dict (dict): Dictionary containing keys for each score, and value of a list with regressor and classifier scores. (Default= 600)

    Returns:
        None

    """

    # Define meta variables
    dataset = meta['name']
    splitting = meta['splitting']
    kfolds = meta['k_folds']
    sample_size = meta['sample_size']
    noise_level = meta['Noise Level']
    name = meta['Algorithm']

    # Define object to be saved to PKL
    pkl_data = [meta, score_dict]

    # Save PKL file
    uniq_tag = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    pklfilename = f'{dataset}_{splitting}_kfolds_{kfolds}_{sample_size}_{noise_level}_{name}_{uniq_tag}.pkl'                            
    pklfile = os.path.join(pkl_path, pklfilename)
    with open(pklfile, 'wb') as f:
        pickle.dump(pkl_data, f)

    return None

def save_pkl_ssh(score_dict = None, meta = None, pkl_path = None):
    """ Save PKL data on a remote server. """

    # Load dotenv
    load_dotenv(dotenv_path=r'C:\Users\skolmar\PycharmProjects\Dichotomization\credentials.env')

    # Define server variables
    host = "cu.epa.gov"
    port = 22
    username = os.environ.get('USER')
    password = os.environ.get('PASSWORD')
    
    # Define meta variables
    dataset = meta['name']
    testset = meta['test_set']
    splitting = meta['splitting']
    sample_size = meta['sample_size']
    noise_level = meta['Noise Level']
    perc = meta['Percentile']
    kfolds = meta['k_folds']
    name = meta['Algorithm']

    # Define object to be saved to PKL
    pkl_data = [meta, score_dict]

    # Prepare PKL file name
    uniq_tag = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    pklfilename = f'{dataset}_{testset}_{splitting}_{sample_size}_{noise_level}_{perc}_{kfolds}_{name}_{uniq_tag}.pkl'                            
    pklfile = os.path.join(pkl_path, pklfilename)

    # Open SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)
    ftp_client = ssh.open_sftp()

    # Save PKL on server
    file = ftp_client.file(pklfile, 'w')
    pickle.dump(pkl_data, file)
    file.flush()

    # Close connections
    ftp_client.close()
    ssh.close()

    return None
###############################
# DNN MODEL BUILDERS
###############################

def reg_dnn_model_builder(hp):
    """ Builds and compiles a regressor DNN model"""

    # Define sequential model
    model = keras.Sequential()

    # Define unit optimization
    if hp is not None:
        hp_units = hp.Int('units', min_value=32, max_value=512, step=64)
    else:
        hp_units = 32

    # Define layer optimization
    if hp is not None:
        hp_layers = hp.Int('n_layers', 2, 8)
    else:
        hp_layers = 5
    
    for i in range(hp_layers):
        model.add(
            layers.Dense(
            units=hp_units,
            kernel_regularizer='l2',
            activation='relu'
            )
        )
        model.add(
            layers.Dropout(0.5)
        )
    
    # Define output layer
    model.add(layers.Dense(1))

    # Define learning rate optimization
    if hp is not None:
        hp_learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001, 0.00001])
    else:
        hp_learning_rate = 0.001

    # Compile model
    model.compile(
        optimizer=optimizers.Adagrad(learning_rate=hp_learning_rate),
        loss='root_mean_squared_error',
        metrics=[metrics.RootMeanSquaredError(), tfa.metrics.RSquare(y_shape=(1,))]
        )

    return model
        
def class_dnn_model_builder(hp, output_bias):
    """ Builds and compiles a classification DNN model"""
    
    # Transform output bias
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    
    # Define unit optimization
    if hp is not None:
        hp_units = hp.Int('units', min_value=32, max_value=512, step=64)
    else:
        hp_units = 32

    # Define sequential model
    model = keras.Sequential()

    # Define layer optimizations
    if hp is not None:
        hp_layers = hp.Int('n_layers', 2, 8)
    else:
        hp_layers = 5
    
    # Loop to create layers
    for i in range(hp_layers):
        model.add(
            layers.Dense(
            units=hp_units,
            kernel_regularizer='l2',
            activation='relu'
            )
        )
        model.add(
            layers.Dropout(0.5)
        )
    
    # Define output layer
    model.add(
        layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    )

    # Define learning rate optimization
    if hp is not None:
        hp_learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001, 0.00001])
    else:
        hp_learning_rate = 0.001

    # Compile model
    model.compile(
        loss=losses.BinaryCrossentropy(),
        optimizer=optimizers.Adagrad(learning_rate=hp_learning_rate),
        metrics=[
            metrics.BinaryAccuracy(name='binary_accuracy'),
            tfa.metrics.F1Score(name='f1_score', num_classes = 2),
            tfa.metrics.CohenKappa(name='cohen_kappa', num_classes = 2),
            tfa.metrics.MatthewsCorrelationCoefficient(name='matthews', num_classes = 2),
            metrics.AUC(name='AUC'),
            ]
        )

    return model