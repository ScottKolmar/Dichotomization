import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import random

from Features.Modeling import two_binner

# We want to show that categorizing an originally continuous target variable
# lowers the observed correlation between variables

# Sample y variable from: Normal distribution, Skewed distribution, Bimodal distribution,
# also varying the continuity in order to make the underlying variable categorical

# Define target variable range
target_variable = np.linspace(0, 1, 200)
loc1, scale1, size1 = (0.2, 0.05, 100)
loc2, scale2, size2 = (0.8, 0.1, 100)
x2 = np.concatenate([np.random.normal(loc=loc1, scale=scale1, size=size1),
                     np.random.normal(loc=loc2, scale=scale2, size=size2)])
bimodal_pdf = stats.norm.pdf(target_variable, loc=loc1, scale=scale1) * float(size1) / x2.size + \
              stats.norm.pdf(target_variable, loc=loc2, scale=scale2) * float(size2) / x2.size

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(target_variable, bimodal_pdf, 'r--', label="bimodal")
ax.plot(target_variable, stats.norm.pdf(target_variable, loc=0.5, scale=0.05), 'b--', label='normal')
ax.plot(target_variable, stats.gamma.pdf(target_variable, a=0.5, loc = 0.2, scale = 2), 'g--', label='gamma')
ax.plot(target_variable, stats.beta.pdf(target_variable, 0.5, 0.5), 'y--', label='beta')
ax.legend(loc=2)
ax.set_xlabel('Target Variable')
ax.set_ylabel('Density')


# Sample from each y-distribution
y_norm = stats.norm.rvs(loc = 0.5, scale=0.05, size=50)
y_bimodal = np.concatenate([stats.norm.rvs(loc1, scale1, size1), stats.norm.rvs(loc2, scale2, size2)])

corrs = []
bin_vect = np.vectorize(two_binner)
corrs_class = []
x_pop = stats.gamma.rvs(a=10, loc = 0.2, scale = 2, size = 10**6)
y_pop = stats.norm.rvs(loc=0.5, scale=0.05, size = 10**6)
pop_corr, pop_p = spearmanr(x_pop, y_pop)
y_pop_class = bin_vect(y_pop, thresh = np.median(y_pop))
pop_class_corr, pop_p_class = stats.pointbiserialr(x_pop, y_pop_class)
pop_class_corr_pear, pop_p_class_pear = stats.pearsonr(x_pop, y_pop_class)
print(f"pop corr: {pop_corr}")
print(f"pop class point biserial corr: {pop_class_corr}")
print(f"pop class pearson r: {pop_class_corr_pear}")
for rep in range(1000):
    x_samples = []
    y_samples = []
    for i in range(50):
        ix = random.randint(0, 10**6)
        x_samples.append(x_pop[ix])
        y_samples.append(y_pop[ix])
    corr, p = spearmanr(x_samples, y_samples)
    corrs.append(corr)
    y_class = bin_vect(y_samples, thresh = np.median(y_pop))
    corr_class, p_class = spearmanr(x_samples, y_class)
    corrs_class.append(corr_class)

print(f"Cont 95%: = {np.percentile(corrs, 0.025), np.percentile(corrs, 0.975)}")
print(f"Class 95%: = {np.percentile(corrs_class, 0.025), np.percentile(corrs_class, 0.975)}")

