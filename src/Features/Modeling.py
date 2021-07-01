# Import Packages
import numpy as np
import pandas as pd
import pickle
import os
from itertools import compress
import random
import datetime

# Import sklearn
from scipy.stats.stats import pearsonr, ttest_ind
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
        
        # Set preprocessing variables to false
        self.var_filter_ = {'filtered': False, 'Value': None}
        self.random_features_ = {'filtered': False, 'Value': None}
        self.drop_corr_ = {'filtered': False, 'Value': None}
        self.scaled_ = False
        self.select_k_best_ = {'filtered': False, 'Value': None}

        return None
    
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

    def generate_data(self, tups=None, k_folds= 5, splitting='Stratified', test_set = 'True'):
        """
        Loops through each y column stored in y_dict (generated from sampleNoise()), generates 9
        binary splits on percentiles 10 through 90, splits each binary split into 10 Folds by Stratified
        KFold(), and trains a classifier/regressor pair on each fold. Takes the average BA score for each fold
        for classifier, turns the continuous regression predictions into classes, and takes average BA score
        from those. Saves the data in a PKL file.

        Parameters:
            tups (list of tuples): List of tuples containing classifier, regressor, name for several algorithms.
            dataset (DataSet instance): DataSet class object.
            k_folds (int): Number of folds used in KFold splitting. (Default = 5)
            splitting (str): 'Stratified' gives StratifiedKFold splitting, and 'Normal' gives KFold splitting. (Default = 'Stratified')
            test_set (str): Can be 'True' or 'Noise'. Determines whether the test set for modeling will have noise or not. (Default = 'True')
        Returns:

        """
        # Set class variables from function inputs
        self.test_set = test_set
        self.splitting = splitting
        self.k_folds = k_folds

        # Set function variables from dataset class variables
        y_dict = self.y_dict
        X = self.X
        y_true = self.y_true
        dataset_name = self.name
        sample_size = self.sample_size
        num_features = self.num_features


        for lvl_dict in y_dict.keys():
            y = y_dict[lvl_dict]['y']
            sigma = y_dict[lvl_dict]['sigma']
            noise_level = str(lvl_dict)

            # Loop through classifier/regressor pairs in tups
            for clf, rgr, alg_name in tups:

                # Define Loop for thresholds
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

                    # meta = make_meta(dataset=dataset_name,
                    #                 sample_size=sample_size,
                    #                 thresh=thresh,
                    #                 perc=perc,
                    #                 algorithm=alg_name,
                    #                 noise_level=noise_level,
                    #                 sigma=sigma,
                    #                 k_folds=k_folds,
                    #                 splitting=splitting,
                    #                 testset=testset,
                    #                 estimator=estimator)

                    # Make a k-fold split and get a classifier and regressor score for each split
                    BA = get_clf_rgr_scores(meta=meta, thresh = thresh, k_folds = k_folds, X=X, y=y, y_true=y_true, clf=clf, rgr=rgr, splitting=splitting)

                    # Save PKL file
                    save_pkl(BA=BA, parent_path = self.parent_path)

                    print('Dataset: {}\n Noise Level: {}\n Split: {}\n Variables: {}\n'.format(dataset_name,
                                                                                            noise_level,
                                                                                            perc,
                                                                                            num_features))
                    for key in BA[1].keys():
                        print('{} Classifier {}: {} +/- {}'.format(meta['Algorithm'],
                                                                key,
                                                                np.average(BA[1][key][0]['Clfs']),
                                                                np.std(BA[1][key][0]['Clfs'])))
                        print('{} Regressor {}: {} +/- {}'.format(meta['Algorithm'],
                                                                key,
                                                                np.average(BA[1][key][1]['Rgrs']),
                                                                np.std(BA[1][key][1]['Rgrs'])))
                    print('\n')

        return None

    def scale_x(self):
        """

        Uses StandardScaler() to scale X data of the dataset.
        """
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

        # Vectorize binning function
        twobin_v = np.vectorize(two_binner)

        # Define y_class and compute class stats
        thresh = np.median(self.y_true)
        y_class = twobin_v(self.y_true, thresh=thresh)
        f_stats_class, p_vals_class = f_classif(self.X,y_class)
        m_i_class = mutual_info_classif(self.X, y_class)

        # Create dataframe
        df_stats = pd.DataFrame(data = {
            'f_stats': f_stats,
            'p_vals': p_vals,
            'm_i': m_i,
            'f_stats_class': f_stats_class,
            'p_vals_class': p_vals_class,
            'm_i_class': m_i_class
            },
            index = self.X.columns)

        return df_stats


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

    scores = {'BA': [dict_BA_clfs, dict_BA_rgrs],
              'F1': [dict_F1_clfs, dict_F1_rgrs],
              'ROC-AUC': [dict_ROCAUC_clfs, dict_ROCAUC_rgrs],
              'Brier': [dict_Brier_clfs, dict_Brier_rgrs],
              'Kappa': [dict_Kappa_clfs, dict_Kappa_rgrs],
              'Logloss': [dict_Logloss_clfs, dict_Logloss_rgrs],
              'Pearsphi': [dict_Pearsphi_clfs, dict_Pearsphi_rgrs]}

    BA = [meta, scores]

    # Loop through splits
    for train_index, test_index in skf.split(X, y_class):
        # Get training and test sets from indices
        X_train = X.iloc[train_index, :]
        y_train = y.iloc[train_index]
        y_train_class = y_class.iloc[train_index]
        X_test = X.iloc[test_index, :]
        y_test_class = y_true_class.iloc[test_index]

        # Fit and predict with clf and get scores
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

def save_pkl(BA=None, parent_path = None):
    """
    Takes a list BA which contains a meta-dictionary and algorithm metrics,
    and saves a PKL file with all the information.

    Parameters:
        BA (list): List containing meta dictionary, dictionary of classifier metrics, and
         dictionary of regressor metrics. (Default= 600)

    Returns:
        600

    """

    # Load dictionary from list
    meta = BA[0]

    # Define meta variables
    dataset = meta['name']
    testset = meta['test_set']
    splitting = meta['splitting']
    sample_size = meta['sample_size']
    noise_level = meta['Noise Level']
    perc = meta['Percentile']
    kfolds = meta['k_folds']
    name = meta['Algorithm']

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
    pklfilename = '{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(dataset,
                                                    testset,
                                                    splitting,
                                                    sample_size,
                                                    noise_level,
                                                    perc,
                                                    kfolds,
                                                    name,
                                                    uniq_tag)
    pklfile = os.path.join(foldpath, pklfilename)
    with open(pklfile, 'wb') as f:
        pickle.dump(BA, f)

    return
