#imports
from scipy.sparse import data
from Features.Modeling import *


csv_file = r'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\Datasets\Solv_desc.csv'
parent_path = r'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\Experiment PKL\Solv\1000_Scaled_VarFilt50_CorrFilt95_OptCont_Random'
dataset = DataSet(csv_file)
dataset.load_dataframe(sample_size = None, num_noise_levels=1, parent_path=parent_path)
var_thresh = np.percentile(dataset.X.var(), 50)
dataset.drop_low_variance_features(var_thresh)
dataset.scale_x()
dataset.drop_correlated_features(0.95)
print(dataset.num_features)
dataset.k_fold_split_train_test(5, 'Random')
dataset.make_algs()
dataset.generate_data(optimize_on = 'Continuous')
