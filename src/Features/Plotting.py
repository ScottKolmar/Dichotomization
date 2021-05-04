import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from src.Features.Modeling import *

# Functions
def metaMatch(file=None, metasearch=None):
    """
    Matches a search meta dictionary with a meta dictionary from a PKL file, and returns True if everything matches.

    Parameters:
           file (PKL): PKL file to load a meta dictionary from.
           metasearch (dict): Meta search dictionary with the same keys as the meta dictionaries in this library.

    Returns:
           True or False

    """
    meta_dict = pickle.load(open(file, 'rb'))[0]
    for k, v in metasearch.items():
        if metasearch[k] == meta_dict[k]:
            pass
        elif metasearch[k] != meta_dict[k]:
            return False

    return True

def getFileString(file):
    """
    Gets file string from PKL file.

    Parameters:
           file (PKL): Absolute filepath for PKL File containing meta dictionary.

    Returns:
           pngstring (str): File string made up from the meta dictionary of the PKL file, with PNG extension.
    """
    meta = pickle.load(open(file, 'rb'))
    meta_dict = meta[0]
    lst = [meta_dict['Dataset'],
           meta_dict['Test Set'],
           meta_dict['Splitting'],
           meta_dict['Sample Size'],
           meta_dict['Noise Level'],
           meta_dict['Percentile'],
           meta_dict['K Folds'],
           meta_dict['Algorithm']]

    lststrs = [str(x) for x in lst]
    pngstring = '_'.join(lststrs) + '.png'

    return pngstring

def loopCols(y_cols_df=None, y_true_class=None, perc = None, sumdict=None):
    """
    For a single cutpoint, loops through each noise columns in a dataframe of noise columns, generates a pandas series
    of boolean values where True means an entry matches the entry for the true class value, and False
    means an entry does not match the entry for the true class value. Each noise level has a Series
    and the Series' are made into a dataframe with noise levels as columns.

    Parameters:
        y_cols_df (dataframe): Pandas dataframe containing the data with added noise. (Default = None).
        y_true_class (Series): Pandas series containing true class values. (Default = None).
        perc (int): Percentile for cutpoint. (Default = None).
        sumdict (dict): Dictionary of boolean sums. Each key is a noise level and each entry
                        is the boolean sum for cutpoints (10 through 90). (Default = None)

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

def plot2D(rootdir = None, metasearch = None):
    """
    Takes a rootdir for PKL files and makes 2D plots of performance metric on the y-axis versus cutpoint on the x-axis,
    for each sigma or noise_level in the data, for a specific algorithm. The stratified parameter should be True if
    the PKL files used stratifiedKFold CV, and the parameter should be False if the PKL files used KFold CV.

    Parameters:
        rootdir (str): Absolute path to root directory for PKL files for a dataset. (Default = None).
        metasearch (dict): Meta dictionary to filter desired files.

    Returns:
        None

    """
    # Define empty classifier and regressor dataframes
    columns = ['Noise', 'Testset', 'Splitting', 'Sigma', 'Percentile', 'Average', 'Std']
    df_score_dict = {'BA': [pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)],
              'F1': [pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)],
              'ROC-AUC': [pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)],
              'Brier': [pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)],
              'Kappa': [pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)],
              'Logloss': [pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)],
              'Pearsphi': [pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)]}

    # Loop through each PKL and fill out df_score_dict
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if metaMatch(file = os.path.join(subdir, file), metasearch = metasearch):
                df_score_dict = scoreDFs(subdir = subdir, file = file, df_score_dict = df_score_dict)
            else:
                pass

    # Loop through each score in df_score_dict
    for key in df_score_dict.keys():
        df_clf = df_score_dict[key][0]
        df_rgr = df_score_dict[key][1]
        score_name = key

        # Define difference column in classifier dataframe, and positive column for color mapping
        df_clf['Difference'] = df_rgr['Average'] - df_clf['Average']
        df_clf['Positive'] = df_clf['Difference'] > 0

        # Define noise strings for dataframe filter
        noisestrs = ['Noise_{}'.format(x) for x in range(10)]

        # Define parent directory for PNG files
        pngparent = r'C:\Users\skolmar\PycharmProjects\Dichotomization\PNG'

        # Parse through Testset column values
        for testsetvalue in ['Noise', 'True']:
            df_clf_testfilt = df_clf[df_clf['Testset'] == testsetvalue]
            df_rgr_testfilt = df_rgr[df_rgr['Testset'] == testsetvalue]

            # Parse through Splitting column values
            for splittingvalue in ['Stratified', 'Random']:
                df_clf_splitfilt = df_clf_testfilt[df_clf_testfilt['Splitting'] == splittingvalue]
                df_rgr_splitfilt = df_rgr_testfilt[df_rgr_testfilt['Splitting'] == splittingvalue]

                # Parse through noise level column values
                for noisestr in noisestrs:

                    # Filter dataframes
                    df_clf_fig = df_clf_splitfilt[df_clf_splitfilt['Noise'] == noisestr]
                    df_rgr_fig = df_rgr_splitfilt[df_rgr_splitfilt['Noise'] == noisestr]

                    # Make figure
                    fig = plt.figure()
                    plt.errorbar(x=df_clf_fig['Percentile'], y=df_clf_fig['Average'], yerr=df_clf_fig['Std'], label='Clf')
                    plt.errorbar(x=df_rgr_fig['Percentile'], y=df_rgr_fig['Average'], yerr=df_rgr_fig['Std'], label='Rgr')
                    plt.legend()
                    plt.xlabel('Percentile')
                    plt.ylabel(score_name)
                    plt.title(metasearch['Algorithm'] + ' ' + noisestr)

                    # Define directory path for PNG file from PKL file path
                    dataset = rootdir.split('\\')[-1]
                    testsetdir = os.listdir(rootdir)[0]
                    testset = testsetvalue
                    splitdir = os.listdir(os.path.join(rootdir, testsetdir))[0]
                    splitting = splittingvalue
                    sample_size = str(metasearch['Sample Size'])
                    pngfoldpath = os.path.join(pngparent, dataset, testset, splitting, sample_size, noisestr)

                    # Make directory paths if they don't exist
                    if not os.path.exists(pngfoldpath):
                        os.makedirs(pngfoldpath)

                    # Define filename
                    kfolds = metasearch['K Folds']
                    pngfilename = '{}_{}_{}_{}_{}_{}_{}_{}.png'.format(dataset,
                                                                    testset,
                                                                    splitting,
                                                                    sample_size,
                                                                    noisestr,
                                                                    kfolds,
                                                                    score_name,
                                                                    metasearch['Algorithm'])

                    # Save figure as PNG
                    plt.savefig(os.path.join(pngfoldpath, pngfilename))
                    plt.close(fig = fig)

    return

def plot3D(rootdir=None, alg=None):
    """
    Makes a 3D plot of Difference between BA of Classifier and Regressor vs. Sigma vs. Cutpoint Percentile.
    Searches a root directory for PKL files that contain the algorithm of choice and loads data into a plot.

    Parameters:
        rootdir (str): Absolute path to the parent directory for the PKL files.
        alg (str): Algorithm to pull data from for the plot. (Default = None).

    Returns:
        None

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
