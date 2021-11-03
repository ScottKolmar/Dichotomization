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

        # Make list of score keys
        self.score_key_list = ['BA', 'Brier', 'F1', 'Kappa', 'Logloss', 'Pearsphi', 'ROC-AUC']

        # Make new dictionary for averaging
        self.ave_dict = {}
        for thresh in range(10, 100, 10):
            self.ave_dict[thresh] = {}
            for score_key in self.score_key_list:
                self.ave_dict[thresh][score_key] = {'Clfs': [], 'Rgrs': []}
        
        return None

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value
    
    def refined_load_scores(self, pkl_file):
        """ Loads scores from pkl file into score sheet object. """

        # Load data from PKL
        pkl_dict = pickle.load(open(pkl_file, 'rb'))
        self.meta_dict = pkl_dict[0]
        list_of_fold_scores = pkl_dict[1]
        
        # Retrieve scores from PKL data and put into new dictionary for averaging
        for fold_score in list_of_fold_scores:
            for score_key in self.score_key_list:
                for i,thresh_score in enumerate(fold_score):
                    thresh_key = i*10 + 10
                    clf_score = thresh_score[score_key]['Clfs']
                    rgr_score = thresh_score[score_key]['Rgrs']
                    self.ave_dict[thresh_key][score_key]['Clfs'].append(clf_score)
                    self.ave_dict[thresh_key][score_key]['Rgrs'].append(rgr_score)
        
        return None

    def load_scores(self, pklfile):
        BA = pickle.load(open(pklfile, 'rb'))
        meta = BA[0]
        scores = BA[1]

        # Remove key-value pairs from meta dict that cause problems for Pandas
        # del meta['features']
        # del meta['Estimator']
        # del meta['tups']

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
        
        return None
    
    def refined_plot_scores(self, png_path):
        """ Plots scores. """

        for score_key in self.score_key_list:
            x_axis = range(10,100,10)
            nobs_clf = np.array([len(self.ave_dict[thresh_key][score_key]['Clfs']) for thresh_key in x_axis])
            nobs_rgr = np.array([len(self.ave_dict[thresh_key][score_key]['Rgrs']) for thresh_key in x_axis])
            clf_aves = [np.average(self.ave_dict[thresh_key][score_key]['Clfs']) for thresh_key in x_axis]
            rgr_aves = [np.average(self.ave_dict[thresh_key][score_key]['Rgrs']) for thresh_key in x_axis]
            clf_stds = [np.std(self.ave_dict[thresh_key][score_key]['Clfs']) for thresh_key in x_axis]
            rgr_stds = [np.std(self.ave_dict[thresh_key][score_key]['Rgrs']) for thresh_key in x_axis]
            ttest_stat, ttest_ps = ttest_ind_from_stats(mean1=clf_aves, std1=clf_stds, nobs1=nobs_clf, mean2=rgr_aves, std2=rgr_stds, nobs2=nobs_rgr)
            
            # Make plots
            fig, ax = plt.subplots(1,1)
            plt.errorbar(x = range(10,100,10), y=clf_aves, yerr=clf_stds, label = 'Clf')
            plt.errorbar(x = range(10,100,10), y=rgr_aves, yerr=rgr_stds, label = 'Rgr')

            # Annotate with asterisk if p < 0.05
            for i,ttest_p in enumerate(ttest_ps):
                if ttest_p < 0.05:
                    ax.annotate("*", xy=(x_axis[i], clf_aves[i] + clf_stds[i]), xycoords='data', size =12)

            # Set labels
            plt.legend()
            plt.xlabel('Percentile')
            plt.ylabel(f'{score_key} Ave.')
            plt.title(self.meta_dict['Algorithm'])
            
            # Save figure and close
            dataset = self.meta_dict['name']
            splitting = self.meta_dict['splitting']
            kfolds = self.meta_dict['k_folds']
            sample_size = self.meta_dict['sample_size']
            noise_level = self.meta_dict['Noise Level']
            name = self.meta_dict['Algorithm']
            png_file_name = f'{dataset}_{splitting}_kfolds_{kfolds}_{sample_size}_{noise_level}_{score_key}_{name}.png'
            plt.savefig(os.path.join(png_path, png_file_name))
            plt.close()

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

#################################
# SSH METHODS
#################################

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
