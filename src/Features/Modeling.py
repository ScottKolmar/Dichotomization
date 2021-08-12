# Import Packages
import numpy as np
import pandas as pd
import pickle
import os
from itertools import compress
import random
import datetime

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
        self.name = self.csv.split('.')[0].split('_')[0].split('\\')[6]

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
        self.train = None
        self.test = None

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

        # Add Clause for KFold versus StratifiedKFold
        if split_method == 'Stratified':
            skf = StratifiedKFold(n_splits=n_folds)
            skf.get_n_splits(self.X, self.y_true)

        elif split_method == 'Random':
            skf = KFold(n_splits= n_folds)
            skf.get_n_splits(self.X, self.y_true)

        self.training_sets = {'X_train': [], 'y_train': []}
        self.testing_sets = {'X_test': [], 'y_test': []}

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
        self.train = self.df.sample(frac=percentage)
        self.test = self.df.loc[~self.df.index.isin(self.train.index)]

        # Set other conflicting variables
        self.k_folds = None
        self.splitting = None
    
        return None
    
    # DEPRECATED
    def set_k_fold_params(self, k_folds, splitting):
        """
        Sets k_fold and splitting class variables.
        """
        # Raise error if training and test set chosen
        if not self.train.empty or self.test.empty:
            return print("Cannot use kfold splitting and train test split simultaneously.")

        # Set class variables according to function inputs
        self.k_folds = k_folds
        self.splitting = splitting

        # Set conflicting variables
        self.train = None
        self.test = None

        return None
    
    # DEPRECATED
    def select_random_features(self, num_features):
        """

        Selects a random number of feature variables.
        """

        # Select features
        feature = random_x(self.df, num_features)

        # Reassign dataset variables
        self.X = self.df.loc[:, feature]
        self.num_features = len(self.X.columns)
        self.features = self.X.columns
        self.random_features_ = {'filtered': True, 'Value': num_features}

        return None

    def make_meta_func(self):
        """
        Takes all the attributes of a dataset object and puts them into an external meta dictionary.

        Parameters:
            dataset (dataset object): DataSet class object.

        Returns:
            meta_dict (dict): Dictionary containing all the dataset attributes.
        """
        meta_dict = {}
        for key in vars(self):
            if key not in ['csv', 'df', 'X', 'y_true', 'y_dict']:
                meta_dict[key] = vars(self)[key]

        return meta_dict

    def generate_data(self, tups, test_set = 'True'):
        """
        Loops through each y column stored in y_dict (generated from sampleNoise()), generates 9
        binary splits on percentiles 10 through 90, and trains a classifier/regressor pair on each fold. Takes the average BA score for each fold
        for classifier, turns the continuous regression predictions into classes, and takes average BA score
        from those. Saves the data in a PKL file.

        Parameters:
            tups (list of tuples): List of tuples containing classifier, regressor, name for several algorithms.
            test_set (str): Can be 'True' or 'Noise'. Determines whether the test set for modeling will have noise or not. (Default = 'True')
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

            # Loop through classifier/regressor pairs in tups
            for clf, rgr, alg_name in tups:

                # Loop through thresholds
                threshes = [np.percentile(y, x) for x in [10, 20, 30, 40, 50, 60, 70, 80, 90]]
                threshtup = list(zip(threshes, [10, 20, 30, 40, 50, 60, 70, 80, 90]))
                for thresh, perc in threshtup:

                    # Define estimator tuple
                    estimator = [clf, rgr, alg_name]

                    # Define meta dictionary from dataset attributes
                    meta = self.make_meta_func()

                    # Fill in meta dictionary from loop specific variables
                    meta['Threshold'] = thresh
                    meta['Percentile'] = perc
                    meta['Algorithm'] = alg_name
                    meta['Noise Level'] = noise_level
                    meta['Sigma'] = sigma
                    meta['Estimator'] = estimator

                    # Check if training sets are used
                    if self.training_sets:
                        score_dict_list = []

                        # Loop through each fold
                        for i, set in enumerate(self.training_sets['X_train']):
                            X_train = self.training_sets['X_train'][i]
                            y_train = self.training_sets['y_train'][i]
                            X_test = self.testing_sets['X_test'][i]
                            y_test = self.testing_sets['y_test'][i]

                            # Get score for single fold
                            score_dict = get_clf_rgr_scores_test_set(thresh=thresh, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, clf = clf, rgr = rgr)
                            score_dict_list.append(score_dict)
                            
                            # Save single score
                            save_pkl(score_dict = score_dict, meta = meta, parent_path = self.parent_path)

                        # Print statements
                        print(f'Dataset: {dataset_name}\n Noise Level: {noise_level}\n Split: {perc}\n Variables: {num_features}\n')
                        
                        # Create empty dictionary for score reporting
                        accumulated_scores = {}
                        for key in score_dict_list[0].keys():
                            accumulated_scores[key] = {'Clfs': [], 'Rgrs': []}

                        # Accumulate the scores from the individual score dictionaries
                        for score_dict in score_dict_list:
                            for key in score_dict.keys():
                                clf_score = score_dict[key][0]['Clfs']
                                accumulated_scores[key]['Clfs'].append(clf_score)
                                rgr_score = score_dict[key][1]['Rgrs']
                                accumulated_scores[key]['Rgrs'].append(rgr_score)

                            print(f"{meta['Algorithm']} Classifier {key}: {np.average(accumulated_scores[key]['Clfs'])} +/- {np.std(accumulated_scores[key]['Clfs'])}")                                   
                            print(f"{meta['Algorithm']} Regressor {key}: {np.average(accumulated_scores[key]['Rgrs'])} +/- {np.std(accumulated_scores[key]['Rgrs'])}")                                            
                        print('\n')

                    # Original k-fold method
                    else:
                        BA = get_clf_rgr_scores(meta=meta, thresh = thresh, k_folds = k_folds, X=X, y=y, y_true=y_true, clf=clf, rgr=rgr)

                        # Save PKL file
                        save_pkl(BA=BA, parent_path = self.parent_path)

                        # Print statements
                        print(f'Dataset: {dataset_name}\n Noise Level: {noise_level}\n Split: {perc}\n Variables: {num_features}\n')
                        for key in score_dict[1].keys():
                            print(f"{meta['Algorithm']} Classifier {key}: {np.average(BA[1][key][0]['Clfs'])} +/- {np.std(BA[1][key][0]['Clfs'])}")                                   
                            print(f"{meta['Algorithm']} Regressor {key}: {np.average(BA[1][key][1]['Rgrs'])} +/- {np.std(BA[1][key][1]['Rgrs'])}")                                            
                        print('\n')

        return None

    def scale_x(self):
        """

        Uses StandardScaler() to scale X data of the dataset.
        """
        # Give error if train test split already performed
        if not isinstance(self.train, type(None)):
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
        if not isinstance(self.train, type(None)):
            return print("Must drop low variance features before train test split")

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
        if not isinstance(self.train, type(None)):
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
    
    def make_opt_algs(self, X, y, optimize_on = ['Continuous', 'Categorical']):
        """

        Optimizes algorithms via GridSearchCV and returns a tuple of optimized clfs and rgrs.

        Inputs:
        optimize_set: Chooses set to optimize hyperparameters on. If set to self.train, will optimize on training set. If set
                        to self.df, will optimize on entire dataset.
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
            'n_estimators': [25, 50, 100, 200, 300, 500],
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

##################################################
# NON CLASS METHODS
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

def make_meta(dataset = None, testset = None, splitting = None, noise_level = None, sigma = None, sample_size = None,
             perc = None, thresh = None, k_folds = None, algorithm = None, estimator = None):
    """
    Makes a meta information dictionary for the PKL file.

    Parameters:
        dataset:
        noise_level:
        sigma:
        sample_size:
        perc:
        thresh:
        k_folds:
        name:
        splitting:

    Returns:
        meta (dict): Dictionary with parameters as values.
    """
    meta = {'Dataset': dataset,
            'Test Set': testset,
            'Splitting': splitting,
            'Noise Level': noise_level,
            'Sigma': sigma,
            'Sample Size': sample_size,
            'Percentile': perc,
            'Threshold': thresh,
            'K Folds': k_folds,
            'Algorithm': algorithm,
            'Estimator': estimator
            }

    return meta

# DEPRECATED
def get_clf_rgr_scores(meta= None, thresh = None, k_folds = None, X= None, y= None, y_true = None, clf = None, rgr = None,
                    splitting = 'Stratified'):
    """
    Takes a meta-dictionary, X, and y, and converts y to classes based on a threshold. Performs Stratified K-Fold
    splitting, fits a classifier and gets a classifier score. Fits a regressor, predicts, converts predictions
    to classes based on threshold, and gets score on those classes. All for a single algorithm (clf/rgr pair).

    Parameters:
        meta (dict): Python dictionary containing meta-information. (Default = 600).
        X (Pandas df slice): Featureset of a descriptor Pandas dataframe. (Default = 600).
        y (Pandas df slice): Continuous y variable of a Pandas dataframe, with added error. (Default = 600).
        y_true (Pandas series): Continuous y variable of true values (no error added). (Default = 600).
        splitting (str): 'Stratified' gives StratifiedKFold splitting, and 'Normal' gives KFold splitting.
                        (Default = 'Stratified')

    Returns:
         BA (list): List of meta dictionary, classifier score dictionary, and regressor score dictionary.
    """

    # Vectorize binning function
    twobin_v = np.vectorize(two_binner)

    # Define y_class
    y_class = pd.DataFrame(twobin_v(y, thresh=thresh), index= y.index)
    y_true_class = pd.DataFrame(twobin_v(y_true, thresh = thresh), index= y_true.index)

    # Add Clause for KFold versus StratifiedKFold
    if splitting == 'Stratified':
        skf = StratifiedKFold(n_splits=k_folds)
        skf.get_n_splits(X, y_class)
    elif splitting == 'Normal':
        skf = KFold(n_splits= k_folds)
        skf.get_n_splits(X, y_class)

    # Define empty lists and dictionaries
    BA_clfs = []
    dict_BA_clfs = {'Clfs': BA_clfs}
    BA_rgrs = []
    dict_BA_rgrs = {'Rgrs': BA_rgrs}

    F1_clfs = []
    dict_F1_clfs = {'Clfs': F1_clfs}
    F1_rgrs = []
    dict_F1_rgrs = {'Rgrs': F1_rgrs}

    ROCAUC_clfs = []
    dict_ROCAUC_clfs = {'Clfs': ROCAUC_clfs}
    ROCAUC_rgrs = []
    dict_ROCAUC_rgrs = {'Rgrs': ROCAUC_rgrs}

    Brier_clfs = []
    dict_Brier_clfs = {'Clfs': Brier_clfs}
    Brier_rgrs = []
    dict_Brier_rgrs = {'Rgrs': Brier_rgrs}

    Kappa_clfs = []
    dict_Kappa_clfs = {'Clfs': Kappa_clfs}
    Kappa_rgrs = []
    dict_Kappa_rgrs = {'Rgrs': Kappa_rgrs}

    Logloss_clfs = []
    dict_Logloss_clfs = {'Clfs': Logloss_clfs}
    Logloss_rgrs = []
    dict_Logloss_rgrs = {'Rgrs': Logloss_rgrs}

    Pearsphi_clfs = []
    dict_Pearsphi_clfs = {'Clfs': Pearsphi_clfs}
    Pearsphi_rgrs = []
    dict_Pearsphi_rgrs = {'Rgrs': Pearsphi_rgrs}

    scores = {
        'BA': [dict_BA_clfs, dict_BA_rgrs],
        'F1': [dict_F1_clfs, dict_F1_rgrs],
        'ROC-AUC': [dict_ROCAUC_clfs, dict_ROCAUC_rgrs],
        'Brier': [dict_Brier_clfs, dict_Brier_rgrs],
        'Kappa': [dict_Kappa_clfs, dict_Kappa_rgrs],
        'Logloss': [dict_Logloss_clfs, dict_Logloss_rgrs],
        'Pearsphi': [dict_Pearsphi_clfs, dict_Pearsphi_rgrs]
        }

    BA = [meta, scores]

    # Loop through splits
    for train_index, test_index in skf.split(X, y_class):
        # Get training and test sets from indices
        X_train = X.iloc[train_index, :]
        y_train = y.iloc[train_index]
        y_train_class = y_class.iloc[train_index]
        X_test = X.iloc[test_index, :]
        y_test_class = y_true_class.iloc[test_index]

        # Fit Clf
        clf.fit(X_train, y_train_class.values.ravel())

        # Classifier probabilistic scores
        y_prob_clf = clf.predict_proba(X_test)
        roc_score_clf = roc_auc_score(y_test_class, y_prob_clf[:,1])
        brier_score_clf = brier_score_loss(y_test_class, y_prob_clf[:,1])
        logloss_score_clf = log_loss(y_test_class, y_prob_clf[:,1])

        # Classifier append probabilistic scores
        ROCAUC_clfs.append(roc_score_clf)
        Brier_clfs.append(brier_score_clf)
        Logloss_clfs.append(logloss_score_clf)

        # Classifier regular scores
        y_pred_clf = clf.predict(X_test)
        BA_score_clf = balanced_accuracy_score(y_test_class, y_pred_clf)
        F1_score_clf = f1_score(y_test_class, y_pred_clf)
        kappa_score_clf = cohen_kappa_score(y_test_class, y_pred_clf)
        pearsphi_score_clf = matthews_corrcoef(y_test_class, y_pred_clf)

        # Classifier append regular scores
        BA_clfs.append(BA_score_clf)
        F1_clfs.append(F1_score_clf)
        Kappa_clfs.append(kappa_score_clf)
        Pearsphi_clfs.append(pearsphi_score_clf)

        # Fit and predict with rgr and get score
        log = LogisticRegression(max_iter= 1000, solver = 'liblinear')
        rgr.fit(X_train, y_train.ravel())
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

        # Append rgr probabilistic scores
        ROCAUC_rgrs.append(roc_score_rgr)
        Brier_rgrs.append(brier_score_rgr)
        Logloss_rgrs.append(logloss_score_rgr)

        # Get rgr regular scores
        BA_score_rgr = balanced_accuracy_score(y_test_class, y_pred_rgr_class)
        F1_score_rgr = f1_score(y_test_class, y_pred_rgr_class)
        kappa_score_rgr = cohen_kappa_score(y_test_class, y_pred_rgr_class)
        pearsphi_score_rgr = matthews_corrcoef(y_test_class, y_pred_rgr_class)

        # Append rgr regular scores
        BA_rgrs.append(BA_score_rgr)
        F1_rgrs.append(F1_score_rgr)
        Kappa_rgrs.append(kappa_score_rgr)
        Pearsphi_rgrs.append(pearsphi_score_rgr)

    return BA

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
    y_test_class = pd.DataFrame(twobin_v(y_test, thresh=thresh), index=y_test.index)

    # Define empty lists and dictionaries
    BA_clfs = []
    BA_rgrs = []

    F1_clfs = []
    F1_rgrs = []

    ROCAUC_clfs = []
    ROCAUC_rgrs = []

    Brier_clfs = []
    Brier_rgrs = []

    Kappa_clfs = []
    Kappa_rgrs = []

    Logloss_clfs = []
    Logloss_rgrs = []

    Pearsphi_clfs = []
    Pearsphi_rgrs = []

    score_dict = {
        'BA': {'Clfs': BA_clfs, 'Rgrs': BA_rgrs},
        'F1': {'Clfs': F1_clfs, 'Rgrs': F1_rgrs},
        'ROC-AUC': {'Clfs': ROCAUC_clfs, 'Rgrs': ROCAUC_rgrs},
        'Brier': {'Clfs': Brier_clfs, 'Rgrs': Brier_rgrs},
        'Kappa': {'Clfs': Kappa_clfs, 'Rgrs': Kappa_rgrs},
        'Logloss': {'Clfs': Logloss_clfs, 'Rgrs': Logloss_rgrs},
        'Pearsphi': {'Clfs': Pearsphi_clfs, 'Rgrs': Pearsphi_rgrs}
        }

    # Fit Clf
    clf.fit(X_train, y_train_class.values.ravel())

    # Classifier probabilistic scores
    y_prob_clf = clf.predict_proba(X_test)
    roc_score_clf = roc_auc_score(y_test_class, y_prob_clf[:,1])
    brier_score_clf = brier_score_loss(y_test_class, y_prob_clf[:,1])
    logloss_score_clf = log_loss(y_test_class, y_prob_clf[:,1])

    # Classifier append probabilistic scores
    ROCAUC_clfs.append(roc_score_clf)
    Brier_clfs.append(brier_score_clf)
    Logloss_clfs.append(logloss_score_clf)

    # Classifier regular scores
    y_pred_clf = clf.predict(X_test)
    BA_score_clf = balanced_accuracy_score(y_test_class, y_pred_clf)
    F1_score_clf = f1_score(y_test_class, y_pred_clf)
    kappa_score_clf = cohen_kappa_score(y_test_class, y_pred_clf)
    pearsphi_score_clf = matthews_corrcoef(y_test_class, y_pred_clf)

    # Classifier append regular scores
    BA_clfs.append(BA_score_clf)
    F1_clfs.append(F1_score_clf)
    Kappa_clfs.append(kappa_score_clf)
    Pearsphi_clfs.append(pearsphi_score_clf)

    # Fit and predict with rgr and get score
    log = LogisticRegression(max_iter= 1000, solver = 'liblinear')
    rgr.fit(X_train, y_train.ravel())
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

    # Append rgr probabilistic scores
    ROCAUC_rgrs.append(roc_score_rgr)
    Brier_rgrs.append(brier_score_rgr)
    Logloss_rgrs.append(logloss_score_rgr)

    # Get rgr regular scores
    BA_score_rgr = balanced_accuracy_score(y_test_class, y_pred_rgr_class)
    F1_score_rgr = f1_score(y_test_class, y_pred_rgr_class)
    kappa_score_rgr = cohen_kappa_score(y_test_class, y_pred_rgr_class)
    pearsphi_score_rgr = matthews_corrcoef(y_test_class, y_pred_rgr_class)

    # Append rgr regular scores
    BA_rgrs.append(BA_score_rgr)
    F1_rgrs.append(F1_score_rgr)
    Kappa_rgrs.append(kappa_score_rgr)
    Pearsphi_rgrs.append(pearsphi_score_rgr)

    return score_dict

def save_pkl(score_dict = None, meta = None, parent_path = None):
    """
    Takes a list BA which contains a meta-dictionary and algorithm metrics,
    and saves a PKL file with all the information.

    Parameters:
        score_dict (dict): Dictionary containing keys for each score, and value of a list with regressor and classifier scores. (Default= 600)

    Returns:
        600

    """

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

    # Define parent directory and specific directory
    if not parent_path:
        parent_path = r'C:\Users\skolmar\PycharmProjects\Dichotomization\PKL'
        foldpath = os.path.join(parent_path,
                                str(dataset),
                                str(testset),
                                str(splitting),
                                str(sample_size),
                                str(noise_level),
                                str(perc))
    elif parent_path:
        foldpath = os.path.join(r'C:\Users\skolmar\PycharmProjects\Dichotomization\PKL', parent_path)



    # Check if directory path exists and make directory
    if not os.path.exists(foldpath):
        os.makedirs(foldpath)

    # Save PKL file
    uniq_tag = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    pklfilename = f'{dataset}_{testset}_{splitting}_{sample_size}_{noise_level}_{perc}_{kfolds}_{name}_{uniq_tag}.pkl'
                                                    
    pklfile = os.path.join(foldpath, pklfilename)
    with open(pklfile, 'wb') as f:
        pickle.dump(pkl_data, f)

    return
