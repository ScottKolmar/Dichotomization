import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from src.Features.Modeling import *

# Functions
class scoreSheet():
    """
    Object for storing scores, used for plotting.

    """

    def __init__(self):
        scorenames = ['BA', 'F1', 'ROC-AUC', 'Brier', 'Kappa', 'Logloss', 'Pearsphi']
        for name in scorenames:
            # Store scores as dataframes inside of a dictionary
            setattr(self, name, {'clf': pd.DataFrame(),
                                 'rgr': pd.DataFrame()})

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

    def load_scores(self, pklfile):
        BA = pickle.load(open(pklfile, 'rb'))
        meta = BA[0]
        scores = BA[1]

        # Iterate through each score
        for key in scores.keys():
            clfs = scores[key][0]['Clfs']
            rgrs = scores[key][1]['Rgrs']

            # Get averages and stds
            clf_ave = np.average(clfs)
            clf_std = np.std(clfs)
            rgr_ave = np.average(rgrs)
            rgr_std = np.std(rgrs)

            # Make dataframes from meta and add scores
            clf_data = pd.DataFrame.from_dict(data=meta, orient='columns')
            clf_data['Average'] = clf_ave
            clf_data['Std'] = clf_std
            clf_data['Score Name'] = key

            rgr_data = pd.DataFrame.from_dict(data=meta, orient='columns')
            rgr_data['Average'] = rgr_ave
            rgr_data['Std'] = rgr_std
            rgr_data['Score Name'] = key

            # Append rows to prior dataframes
            self.__getattribute__(str(key))['clf'] = self.__getattribute__(str(key))['clf'].append(clf_data, ignore_index=True)
            self.__getattribute__(str(key))['rgr'] = self.__getattribute__(str(key))['rgr'].append(rgr_data, ignore_index=True)

def plot_scores(rootdir, search_dict = None):
    """

    Parameters:
    rootdir (str): Absolute filepath to the directory which contains the PKL files of interest,
                    i.e. '...\\PKL\\g298atom'. The end of the path will be used to assign the dataset in the output
                    PNG file.
    search_dict (dict): Dictionary for finding the PKL files of interest. The keys for the dictionary must be found in
                        the meta_dictionary of the PKL files. For example, 'Algorithm', 'Noise', 'Testset', etc. This
                        dictionary will filter the files for building the plots.

    Returns:

    :return:
    """
    # Initialize scoreSheet object
    sheet = scoreSheet()

    # Loop through each PKL and fill out df_score_dict
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            pklfile = os.path.join(subdir, file)
            sheet.load_scores(pklfile)

    # Filtering
    for key in vars(sheet).keys():
        df_clf = sheet.__getattribute__(str(key))['clf']
        df_rgr = sheet.__getattribute__(str(key))['rgr']
        if search_dict:
            for search_key in search_dict.keys():
                df_clf = df_clf[df_clf[search_key] == search_dict[search_key]]
                df_rgr = df_rgr[df_rgr[search_key] == search_dict[search_key]]
        else:
            pass

        # Print error statement if empty df
        if df_clf.empty:
            print('Filtering has produced an empty classifier dataframe.')
        elif df_rgr.empty:
            print('Filtering has produced an empty regressor dataframe.')
        else:
            # Make figure
            fig = plt.figure()
            plt.errorbar(x=df_clf['Percentile'], y=df_clf['Average'], yerr=df_clf['Std'], label='Clf')
            plt.errorbar(x=df_rgr['Percentile'], y=df_rgr['Average'], yerr=df_rgr['Std'], label='Rgr')
            plt.legend()
            plt.xlabel('Percentile')
            plt.ylabel(key)
            plt.title(str(df_clf['Algorithm'][0]) + ' ' + str(df_clf['Noise Level'][0]))

            # Define directory path for PNG file from PKL file path
            pngparent = r'C:\Users\skolmar\PycharmProjects\Dichotomization\PNG'
            dataset = str(df_clf['Dataset'][0])
            testset = str(df_clf['Test Set'][0])
            splitting = str(df_clf['Splitting'][0])
            sample_size = str(df_clf['Sample Size'][0])
            noise = str(df_clf['Noise Level'][0])
            pngfoldpath = os.path.join(pngparent, dataset, testset, splitting, sample_size, noise)

            # Make directory paths if they don't exist
            if not os.path.exists(pngfoldpath):
                os.makedirs(pngfoldpath)

            # Define filename
            pngfilename = '{}_{}_{}_{}_{}_{}_{}_{}.png'.format(dataset,
                                                               testset,
                                                               splitting,
                                                               sample_size,
                                                               noise,
                                                               df_clf['K Folds'][0],
                                                               df_clf['Score Name'][0],
                                                               df_clf['Algorithm'][0])

            # Save figure as PNG
            plt.savefig(os.path.join(pngfoldpath, pngfilename))
            plt.close(fig=fig)

    return sheet

def loopCols(y_cols_df=None, y_true_class=None, perc = None, sumdict=None):
    """
    For a single cutpoint, loops through each noise columns in a dataframe of noise columns, generates a pandas series
    of boolean values where True means an entry matches the entry for the true class value, and False
    means an entry does not match the entry for the true class value. Each noise level has a Series
    and the Series' are made into a dataframe with noise levels as columns.

    Parameters:
        y_cols_df (dataframe): Pandas dataframe containing the data with added noise. (Default = 600).
        y_true_class (Series): Pandas series containing true class values. (Default = 600).
        perc (int): Percentile for cutpoint. (Default = 600).
        sumdict (dict): Dictionary of boolean sums. Each key is a noise level and each entry
                        is the boolean sum for cutpoints (10 through 90). (Default = 600)

    Returns:
        bool_df (dataframe): Pandas dataframe containing booleans for matching y_true_class.

    """
    # Define empty list of boolean sums and add it as a key (percentile) value (list) pair in sumdict
    sums = []
    sumdict[perc] = sums
    bool_df = pd.DataFrame(index=y_cols_df.index)

    # Loop through each noise column in y_col_df
    for col in y_cols_df.columns:
        # Define a boolean column for matching the true class values
        bools = (y_cols_df[col] == y_true_class[0])

        # Add boolean column to empty bool_df
        bool_df[col] = bools

        # Calculate boolean sum and add to sum list
        sum = bools.sum()
        sums.append(sum)

    return bool_df

def makeHeatMaps(y_true=None, y_dict=None):
    """

    :param y_true:
    :param y_dict:
    :return:
    """
    two_bin_v = np.vectorize(twobinner)
    cut = np.percentile(y_true, 50)
    y_true_class = pd.DataFrame(two_bin_v(y_true, cut), index=y_true.index)

    percs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    perdict = {}
    colsdfdict = {}
    sumdict = {}
    bools_df_dict = {}
    for perc in percs:
        cut = np.percentile(y_true, perc)
        y_true_class = pd.DataFrame(two_bin_v(y_true, cut), index=y_true.index)
        perdict[perc] = y_true_class

        y_cols_df = pd.DataFrame()
        colsdfdict[perc] = y_cols_df

        # Define noise strings
        noisestrs = ['Noise_{}'.format(x) for x in range(10)]

        # Make tuples of (noise_level, Noise_x) from y_dict and noise strings
        keytups = list(zip(y_dict.keys(), noisestrs))

        # Loop through each tuple
        for key, noisestr in keytups:
            # Define y_col and add it to y_cols_df
            y_col = y_dict[key]['y']
            y_cols_df[noisestr] = pd.Series(two_bin_v(y_col, cut), index=y_col.index)

            bool_df = loopCols(y_cols_df=y_cols_df, y_true_class=y_true_class, perc=perc, sumdict=sumdict)
            bools_df_dict[perc] = bool_df

    for key in bools_df_dict:
        data = bools_df_dict[key]
        fig = plt.figure()
        ax = sns.heatmap(data)
        pngfolder = r'C:\Users\skolmar\PycharmProjects\Dichotomization\HeatMaps'
        pngfile = '{}_{}_heatmap.png'.format(dataset, key)
        plt.savefig(os.path.join(pngfolder, dataset, pngfile))

    return

def plot3D(rootdir=None, alg=None):
    """
    Makes a 3D plot of Difference between BA of Classifier and Regressor vs. Sigma vs. Cutpoint Percentile.
    Searches a root directory for PKL files that contain the algorithm of choice and loads data into a plot.

    Parameters:
        rootdir (str): Absolute path to the parent directory for the PKL files.
        alg (str): Algorithm to pull data from for the plot. (Default = 600).

    Returns:
        600

    """
    columns = ['Noise', 'Sigma', 'Percentile', 'Average', 'Std']
    df_clf = pd.DataFrame(columns=columns)
    df_rgr = pd.DataFrame(columns=columns)

    # Loop through each PKL for algorithm
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.split('.')[0].split('_')[-1] == alg:
                df_clf, df_rgr = scoreDFs(subdir = subdir, file = file, df_clf = df_clf, df_rgr = df_rgr)

    df_clf['Difference'] = df_rgr['Average'] - df_clf['Average']
    df_clf['Positive'] = df_clf['Difference'] > 0

    # Instantiate 3D figure
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot_trisurf(df_clf['Sigma'], df_clf['Percentile'], df_clf['Difference'], cmap = 'RdBu')
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Cutpoint Percentile')
    ax.set_zlabel('Difference')
    ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90])

    plt.show()

    return
