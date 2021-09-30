import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import scipy.stats
from scipy.stats.stats import ttest_ind_from_stats
from Features.Modeling import *
from datetime import datetime

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

        # Remove key-value pairs from meta dict that cause problems for Pandas
        del meta['features']
        del meta['Estimator']
        del meta['tups']

        # Iterate through each score
        for key in scores.keys():
            clfs = scores[key]['Clfs']
            rgrs = scores[key]['Rgrs']

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
            ttest_stat, ttest_p = ttest_ind_from_stats(mean1=clf_ave, std1=clf_std, nobs1=5, mean2=rgr_ave, std2=rgr_std, nobs2=5)
            rgr_data['TTest p'] = ttest_p

            # Append rows to prior dataframes
            self.__getattribute__(str(key))['clf'] = self.__getattribute__(str(key))['clf'].append(clf_data, ignore_index=True)
            self.__getattribute__(str(key))['rgr'] = self.__getattribute__(str(key))['rgr'].append(rgr_data, ignore_index=True)

    def load_scores_SSH(self, remote_path, k_fold):
        """ Loads scores from a remote path using SSH.
        Inputs:
        remote_path (string): Absolute path where PKL files reside on the remote server.
        k_fold (int): Index of the desired k_fold to plot, if the models were built using k_fold CV.

        """

        # Load credential file
        load_dotenv(dotenv_path=r'C:\Users\skolmar\PycharmProjects\Dichotomization\credentials.env')

        # Define SSH variables
        host = "cu.epa.gov"
        port = 22
        username = os.environ.get('USER')
        password = os.environ.get('PASSWORD')

        # Load SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, port, username, password)

        # Connect FTP
        sftp_client = ssh.open_sftp()

        # Order remote file list by name
        remote_list = sftp_client.listdir(remote_path)
        remote_list_ordered = sorted(remote_list, key =lambda x:x.split('.')[0])

        time0 = datetime.now()

        # Get number of k_folds
        file_name = os.path.join(remote_path, remote_list_ordered[0])
        BA = pickle.load(sftp_client.file(file_name, 'r'))
        meta = BA[0]
        k_folds = meta['k_folds']

        time1 = datetime.now()
        print(f"{time1-time0}")

        # Catch clause if user provides a k fold index out of range
        if k_fold > k_folds:
            print(f'Number of k_folds is {k_folds}, user must choose a k_fold less than this value.')
            return

        # Loop through every nth file where n is number of folds
        for f in remote_list_ordered[k_fold::k_folds]:
            remote_file_name = os.path.join(remote_path, f)
            BA = pickle.load(sftp_client.file(remote_file_name, 'r'))

            # Define variables
            meta = BA[0]
            scores = BA[1]

            # Remove key-value pairs from meta dict that cause problems for Pandas
            del meta['features']
            del meta['Estimator']
            del meta['tups']

            # Iterate through each score
            for key in scores.keys():
                clfs = scores[key]['Clfs']
                rgrs = scores[key]['Rgrs']

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
                clf_data['Which training k_fold'] = k_fold

                rgr_data = pd.DataFrame.from_dict(data=meta, orient='columns')
                rgr_data['Average'] = rgr_ave
                rgr_data['Std'] = rgr_std
                rgr_data['Score Name'] = key
                rgr_data['Which training k_fold'] = k_fold
                ttest_stat, ttest_p = ttest_ind_from_stats(mean1=clf_ave, std1=clf_std, nobs1=5, mean2=rgr_ave, std2=rgr_std, nobs2=5)
                rgr_data['TTest p'] = ttest_p

                # Append rows to prior dataframes
                self.__getattribute__(str(key))['clf'] = self.__getattribute__(str(key))['clf'].append(clf_data, ignore_index=True)
                self.__getattribute__(str(key))['rgr'] = self.__getattribute__(str(key))['rgr'].append(rgr_data, ignore_index=True)

            time2 = datetime.now()
            print(f"{time2-time0}")

        # Close sftp and ssh connections
        sftp_client.close()
        ssh.close()

        return None
    
    def load_scores_SSH_opttrainsamefold(self, remote_path):
        """
        Loads scores by associating a single score with optimization and training on the same training fold.

        """

        # Load credential file
        load_dotenv(dotenv_path=r'C:\Users\skolmar\PycharmProjects\Dichotomization\credentials.env')

        # Define SSH variables
        host = "cu.epa.gov"
        port = 22
        username = os.environ.get('USER')
        password = os.environ.get('PASSWORD')

        # Load SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, port, username, password)

        # Connect FTP
        sftp_client = ssh.open_sftp()

        # Order remote file list by name
        remote_list = sftp_client.listdir(remote_path)
        remote_list_ordered = sorted(remote_list, key =lambda x:x.split('.')[0])

        # Record time before loop
        time0 = datetime.now()

        # Load each score dictionary into dataframe
        for i,f in enumerate(remote_list_ordered):
            remote_file_name = os.path.join(remote_path, f)
            BA = pickle.load(sftp_client.file(remote_file_name, 'r'))
            meta = BA[0]
            score_dict = BA[1]
            
            # Remove key-value pairs from meta dict that cause problems for Pandas
            del meta['features']
            del meta['Estimator']
            del meta['tups']

            # Obtain the k_fold that was used to optimize hyperparameters
            k_fold_number = i % meta['k_folds']

            # Iterate through each key in single scores dictionary
            for key in score_dict.keys():
                clf_score = score_dict[key]['Clfs'][k_fold_number]
                rgr_score = score_dict[key]['Rgrs'][k_fold_number]

                # Make dataframes from meta and add scores
                clf_data = pd.DataFrame.from_dict(data=meta, orient='columns')
                clf_data['score'] = clf_score
                clf_data['Score Name'] = key
                clf_data['Which training k_fold'] = k_fold_number

                rgr_data = pd.DataFrame.from_dict(data=meta, orient='columns')
                rgr_data['score'] = rgr_score
                rgr_data['Score Name'] = key
                clf_data['Which training k_fold'] = k_fold_number

                # Append rows to prior dataframes
                self.__getattribute__(str(key))['clf'] = self.__getattribute__(str(key))['clf'].append(clf_data, ignore_index=True)
                self.__getattribute__(str(key))['rgr'] = self.__getattribute__(str(key))['rgr'].append(rgr_data, ignore_index=True)

            
            # Time every 5 loops
            if k_fold_number == 4:
                time_i = datetime.now()
                print(f'Loop {i} time: {time_i-time0}')

        # Close sftp and ssh connections
        sftp_client.close()
        ssh.close()

        return None
        
    def plot_scores(self, search_dict = None, png_path = None):
        """ Plots scores. """

        # Load dataframes as function variables for manipulation, leaving class variables unaffected
        for key in vars(self).keys():
            df_clf = self.__getattribute__(str(key))['clf']
            df_rgr = self.__getattribute__(str(key))['rgr']

            # Require that the search_dict have at least a filter for algorithm
            if not search_dict:
                print("User must provide at least a filter for algorithm in search_dict via 'Algorithm': value")
                return

            elif 'Algorithm' not in search_dict.keys():
                print("User must provide at least a filter for algorithm in search_dict via 'Algorithm': value")
                return

            else:

                # Apply each filter in search_dict keys
                for search_key in search_dict.keys():
                    df_clf = df_clf[df_clf[search_key] == search_dict[search_key]]
                    df_rgr = df_rgr[df_rgr[search_key] == search_dict[search_key]]

                # Print error statement if filter results in empty df
                if df_clf.empty:
                    print('Filtering has produced an empty classifier dataframe.')

                elif df_rgr.empty:
                    print('Filtering has produced an empty regressor dataframe.')

                else:
                    
                    # Calculate averages, std, and ttest
                    clf_x = df_clf.groupby(['Percentile']).mean().index
                    clf_ave = df_clf.groupby(['Percentile']).mean()['score']
                    clf_std = df_clf.groupby(['Percentile']).std()['score']
                    rgr_x = df_rgr.groupby(['Percentile']).mean().index
                    rgr_ave = df_rgr.groupby(['Percentile']).mean()['score']
                    rgr_std = df_rgr.groupby(['Percentile']).std()['score']
                    ttest_stat, ttest_p = ttest_ind_from_stats(mean1=clf_ave, std1=clf_std, nobs1=len(clf_x), mean2=rgr_ave, std2=rgr_std, nobs2=len(rgr_x))

                    # Make figure with annotations
                    fig, ax = plt.subplots(1,1)
                    plt.errorbar(x=clf_x, y=clf_ave, yerr=clf_std, label='Clf')
                    plt.errorbar(x=rgr_x, y=rgr_ave, yerr=rgr_std, label='Rgr')

                    # Annotate with asterisk if p < 0.05
                    for i in range(len(clf_x)):
                        if ttest_p[i] < 0.05:
                            ax.annotate("*", xy=(clf_x[i], clf_ave.iloc[i] + clf_std.iloc[i]), xycoords='data', size =12)
                    
                    # Set labels and show figure
                    ax.set_xlabel('X')
                    ax.set_ylabel('y')
                    plt.legend()
                    plt.xlabel('Percentile')
                    plt.ylabel(key)
                    plt.title(search_dict['Algorithm'] + ' ' + str(df_clf['Noise Level'].iloc[0]))

                    # Define directory path for PNG file from PKL file path
                    if not png_path:
                        pngparent = r'C:\Users\skolmar\PycharmProjects\Dichotomization\PNG'
                    else:
                        pngparent = png_path
                    dataset = str(df_clf['name'].iloc[0])
                    testset = str(df_clf['test_set'].iloc[0])
                    splitting = str(df_clf['splitting'].iloc[0])
                    sample_size = str(df_clf['sample_size'].iloc[0])
                    noise = str(df_clf['Noise Level'].iloc[0])
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
                                                                    df_clf['k_folds'].iloc[0],
                                                                    df_clf['Score Name'].iloc[0],
                                                                    search_dict['Algorithm'])

                    # Save figure as PNG
                    plt.savefig(os.path.join(pngfoldpath, pngfilename))
                    plt.close()
        return None


def plot_scores(rootdir, search_dict = None, png_path = None):
    """

    Parameters:
    rootdir (str): Absolute filepath to the directory which contains the PKL files of interest,
                    i.e. '...\\PKL\\g298atom'. The end of the path will be used to assign the dataset in the output
                    PNG file.
    search_dict (dict): Dictionary for finding the PKL files of interest. The keys for the dictionary must be found in
                        the meta_dictionary of the PKL files. For example, 'Algorithm', 'Noise', 'test_set', etc. This
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

    # Load dataframes as function variables for manipulation, leaving class variables unaffected
    for key in vars(sheet).keys():
        df_clf = sheet.__getattribute__(str(key))['clf']
        df_rgr = sheet.__getattribute__(str(key))['rgr']

        # Require that the search_dict have at least a filter for algorithm
        if not search_dict:
            print("User must provide at least a filter for algorithm in search_dict via 'Algorithm': value")
            return

        elif 'Algorithm' not in search_dict.keys():
            print("User must provide at least a filter for algorithm in search_dict via 'Algorithm': value")
            return

        else:

            # Apply each filter in search_dict keys
            for search_key in search_dict.keys():
                df_clf = df_clf[df_clf[search_key] == search_dict[search_key]]
                df_rgr = df_rgr[df_rgr[search_key] == search_dict[search_key]]

            # Print error statement if filter results in empty df
            if df_clf.empty:
                print('Filtering has produced an empty classifier dataframe.')

            elif df_rgr.empty:
                print('Filtering has produced an empty regressor dataframe.')

            else:

                # # Make figure
                # fig = plt.figure()
                # plt.errorbar(x=df_clf['Percentile'], y=df_clf['Average'], yerr=df_clf['Std'], label='Clf')
                # plt.errorbar(x=df_rgr['Percentile'], y=df_rgr['Average'], yerr=df_rgr['Std'], label='Rgr')
                # plt.legend()
                # plt.xlabel('Percentile')
                # plt.ylabel(key)
                # plt.title(search_dict['Algorithm'] + ' ' + str(df_clf['Noise Level'].iloc[0]))

                fig, ax = plt.subplots(1,1)
                plt.errorbar(x=df_clf['Percentile'], y=df_clf['Average'], yerr=df_clf['Std'], label='Clf')
                plt.errorbar(x=df_rgr['Percentile'], y=df_rgr['Average'], yerr=df_rgr['Std'], label='Rgr')
                for i in range(len(df_clf['Percentile'])):
                    if df_rgr['TTest p'][i] < 0.05:
                        ax.annotate("*", xy=(df_clf['Percentile'][i], df_clf['Average'][i] + df_clf['Std'][i]), xycoords='data', size =12)
                ax.set_xlabel('X')
                ax.set_ylabel('y')
                plt.legend()
                plt.xlabel('Percentile')
                plt.ylabel(key)
                plt.title(search_dict['Algorithm'] + ' ' + str(df_clf['Noise Level'].iloc[0]))
                plt.show()

                # Define directory path for PNG file from PKL file path
                if not png_path:
                    pngparent = r'C:\Users\skolmar\PycharmProjects\Dichotomization\PNG'
                else:
                    pngparent = png_path
                dataset = str(df_clf['name'].iloc[0])
                testset = str(df_clf['test_set'].iloc[0])
                splitting = str(df_clf['splitting'].iloc[0])
                sample_size = str(df_clf['sample_size'].iloc[0])
                noise = str(df_clf['Noise Level'].iloc[0])
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
                                                                   df_clf['k_folds'].iloc[0],
                                                                   df_clf['Score Name'].iloc[0],
                                                                   search_dict['Algorithm'])

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
