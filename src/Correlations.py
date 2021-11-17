from pandas.core.frame import DataFrame
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from statsmodels.stats.multitest import multipletests
from sklearn.feature_selection import f_regression

from Features.Modeling import DataSet

g298_file = r'C:\Users\skolmar\PycharmProjects\Dichotomization\Datasets\g298atom_desc.csv'
dataset = DataSet(g298_file)
dataset.load_dataframe(sample_size=1000, num_noise_levels=1)
dataset.scale_x()
dataset.drop_low_variance_features()
dataset.drop_correlated_features()
df_stats = dataset.compare_feature_statistics()
print(df_stats)

data_dict = {
    'corr_diff': df_stats['sp_corrs_class'] - df_stats['sp_corrs'],
    'f_stats_diff': df_stats['f_stats_class'] - df_stats['f_stats'],
    'm_i_diff': df_stats['m_i_class'] - df_stats['m_i'],
}
df_diff = pd.DataFrame(data=data_dict)
# df_diff = (df_diff - df_diff.min())/(df_diff.max() - df_diff.min())
plt.bar(df_diff.index, df_diff['corr_diff'].sort_values())
plt.bar(df_diff.index, df_diff['f_stats_diff'].sort_values())
# plt.bar(df_diff.index, df_diff['m_i_diff'].sort_values())
plt.show()

# correlations, uncorrected_p_values = get_correlations(df)

# # Correct p-values for multiple testing and check significance (True if the corrected p-value < 0.05)
# shape = uncorrected_p_values.values.shape
# significant_matrix = multipletests(uncorrected_p_values.values.flatten())[0].reshape(
#     shape
# )

# # Here we start plotting
# g = sns.clustermap(correlations, cmap="vlag", vmin=-1, vmax=1)

# # Here labels on the y-axis are rotated
# for tick in g.ax_heatmap.get_yticklabels():
#     tick.set_rotation(0)

# # Here we add asterisks onto cells with signficant correlations
# for i, ix in enumerate(g.dendrogram_row.reordered_ind):
#     for j, jx in enumerate(g.dendrogram_row.reordered_ind):
#         if i != j:
#             text = g.ax_heatmap.text(
#                 j + 0.5,
#                 i + 0.5,
#                 "*" if significant_matrix[ix, jx] or significant_matrix[jx, ix] else "",
#                 ha="center",
#                 va="center",
#                 color="black",
#             )
#             text.set_fontsize(20)

# # Save a high-res copy of the image to disk
# plt.tight_layout()
# plt.show()