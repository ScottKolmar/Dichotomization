#imports
from scipy.sparse import data
from Features.Modeling import *


csv_file = r'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\Datasets\Solv_desc.csv'
parent_path = r'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\Experiment PKL\Solv\1000_NoPre_OptCont_Random'
dataset = DataSet(csv_file)
dataset.load_dataframe(sample_size = None, num_noise_levels=1, parent_path=parent_path)
dataset.k_fold_split_train_test(5, 'Random')
dataset.make_algs()
dataset.generate_data(optimize_on = 'Continuous')
