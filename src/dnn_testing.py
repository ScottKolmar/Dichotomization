from Features.Modeling import *
from pprint import PrettyPrinter

for kernel_regularizer in [None, 'l1', 'l2']:
    if not kernel_regularizer:
        kernel_string = 'No_Reg'
    else:
        kernel_string = kernel_regularizer
    for n_units in [[32], [64], [128]]:
        for n_hidden_layers in [2,3,4,5,6,7,8]:
            unit_string = n_units[0]
            csv_file = r'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\Datasets\g298atom_desc.csv'
            parent_path = fr'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\Experiment PKL\g298atom\DNNs\{kernel_string}\{unit_string}_units\{n_hidden_layers}hiddenlayers_relu'
            dataset = DataSet(csv_file)
            dataset.load_dataframe(sample_size = 1000, num_noise_levels=1, parent_path=parent_path)
            dataset.scale_x()
            dataset.k_fold_split_train_test(5, 'Random')
            dataset.make_dnns(
                n_hidden_layers=n_hidden_layers,
                n_units=n_units,
                activation='relu',
                kernel_regularizer=kernel_regularizer,
                learning_rate=0.001
                )
            dataset.generate_data(optimize_on=None)


