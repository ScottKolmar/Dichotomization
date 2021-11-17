from Features.Modeling import *
from pprint import PrettyPrinter

# csv_file = r'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\Datasets\g298atom_desc.csv'
# parent_path = fr'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\Experiment PKL\g298atom\DNNs\128_units\1000_Scaled_5layers_128units_relu_Random'
# dataset = DataSet(csv_file)
# dataset.load_dataframe(sample_size = 100000, num_noise_levels=1, parent_path=parent_path)

# X_train = dataset.X.sample(frac=0.8)
# X_test = dataset.X.drop(X_train.index)
# y_train = dataset.y_true[X_train.index]
# y_test = dataset.y_true[X_test.index]

reg_model = reg_dnn_model_builder(n_units=[32], n_hidden_layers=10, kernel_regularizer = None, learning_rate = 0.01)
print(reg_model.layers[1].get_config())