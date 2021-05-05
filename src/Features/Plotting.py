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

def scoreDFs(subdir = None, file = None, df_score_dict = None):
    """
    Appends score information from PKL file to empty score dataframes.

    Parameters:
         subdir ():
         file ():
         df_score_dict ():

    Returns:
        600

    """

    # Load pickle from OS path
    BA = pickle.load(open(os.path.join(subdir, file), 'rb'))
    scores = BA[1]

    for key in scores.keys():
        clfs = scores[key][0]['Clfs']
        rgrs = scores[key][1]['Rgrs']

        # Get averages and stds
        clf_ave = np.average(clfs)
        clf_std = np.std(clfs)
        rgr_ave = np.average(rgrs)
        rgr_std = np.std(rgrs)

        # Load meta and assign values to variables
        meta = BA[0]
        algorithm = meta['Algorithm']
        noise = meta['Noise Level']
        sigma = meta['Sigma']
        perc = meta['Percentile']
        testset = meta['Test Set']
        splitting = meta['Splitting']

        # Make new dictionaries out of meta dictionary
        clf_data = {'Algorithm': [algorithm],
                    'Noise': [noise],
                    'Testset': [testset],
                    'Splitting': [splitting],
                    'Sigma': [sigma],
                    'Percentile': [perc],
                    'Average': [clf_ave],
                    'Std': [clf_std]}
        rgr_data = {'Algorithm': [algorithm],
                    'Noise': [noise],
                    'Testset': [testset],
                    'Splitting': [splitting],
                    'Sigma': [sigma],
                    'Percentile': [perc],
                    'Average': [rgr_ave],
                    'Std': [rgr_std]}

        # Convert to rows in new dataframes
        clf_data_df = pd.DataFrame.from_dict(data=clf_data, orient='columns')
        rgr_data_df = pd.DataFrame.from_dict(data=rgr_data, orient='columns')

        # Append rows to prior dataframes
        df_score_dict[key][0] = df_score_dict[key][0].append(clf_data_df, ignore_index = True)
        df_score_dict[key][1] = df_score_dict[key][1].append(rgr_data_df, ignore_index=True)

    return df_score_dict

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

def plot2D(rootdir = None, metasearch = None):
    """
    Takes a rootdir for PKL files and makes 2D plots of performance metric on the y-axis versus cutpoint on the x-axis,
    for each sigma or noise_level in the data, for a specific algorithm. The stratified parameter should be True if
    the PKL files used stratifiedKFold CV, and the parameter should be False if the PKL files used KFold CV.

    Parameters:
        rootdir (str): Absolute path to root directory for PKL files for a dataset. (Default = 600).
        metasearch (dict): Meta dictionary to filter desired files.

    Returns:
        600

    """
    # Define empty classifier and regressor dataframes
    columns = ['Algorithm', 'Noise', 'Testset', 'Splitting', 'Sigma', 'Percentile', 'Average', 'Std']
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
            #if metaMatch(file = os.path.join(subdir, file), metasearch = metasearch):
            df_score_dict = scoreDFs(subdir = subdir, file = file, df_score_dict = df_score_dict)
            #else:
            #    pass

    # Loop through each score in df_score_dict
    for key in df_score_dict.keys():
        df_clf = df_score_dict[key][0]
        df_rgr = df_score_dict[key][1]
        score_name = key
        algorithm = df_clf.loc[0,'Algorithm']

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
                    plt.title(algorithm + ' ' + noisestr)

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
