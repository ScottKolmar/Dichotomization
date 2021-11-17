from Features.Plotting import *
import os
import shutil

pkl_folder = r'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\Experiment PKL\g298atom\DNNs'
png_folder = r'C:\Users\skolmar\PycharmProjects\Modeling\Dichotomization\Experiment PNG\g298atom\DNNs'

for kernel_regularizer in [None, 'l1', 'l2']:
    if not kernel_regularizer:
        kernel_string = 'No_Reg'
    else:
        kernel_string = kernel_regularizer
    for n_units in [32, 64, 128]:
        for n_hidden_layers in [2,3,4,5,6,7,8]:
            pkl_folder_string = os.path.join(pkl_folder, kernel_string, f'{n_units}_units', f'{n_hidden_layers}hiddenlayers_relu')
            png_folder_string = os.path.join(png_folder, kernel_string, f'{n_units}_units', f'{n_hidden_layers}hiddenlayers_relu')
            for file in os.listdir(pkl_folder_string):
                sheet = scoreSheet()
                sheet.refined_load_scores(pkl_file= os.path.join(pkl_folder_string, file))
                sheet.refined_plot_scores(png_path = png_folder_string)
