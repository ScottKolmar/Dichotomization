#imports
from scipy.sparse import data
from Features.Modeling import *
from sklearn.svm import SVC
from pandas.api.types import infer_dtype
from sklearn.utils import check_X_y
import copy

# Real Code
csv_file = r'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\Datasets\Solv_desc.csv'
parent_path = r'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\Experiment PKL\Solv\1000_Scaled_NoOpt_Random'
dataset = DataSet(csv_file)
dataset.load_dataframe(sample_size = None, num_noise_levels=1, parent_path=parent_path)
dataset.scale_x()
dataset.k_fold_split_train_test(5, 'Random')
dataset.make_algs()
dataset.generate_data(optimize_on=False)
