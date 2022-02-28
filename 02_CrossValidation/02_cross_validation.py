import os

import numpy as np
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt

df = pd.read_csv("../data/winequality-red.csv")
df = df.sample(frac=1).reset_index(drop=True)

df['fold'] = -1

#########################################################
######                RANDOM K FOLD                 #####
#########################################################
kfold = model_selection.KFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X=df)):
    df.loc[val_idx, 'fold'] = fold

if not os.path.exists("./output"):
    os.makedirs("./output")
df.to_csv("./output/train_random_folds.csv")

#########################################################
######              Stratified K Fold               #####
#########################################################

kfold = model_selection.StratifiedKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X=df, y=df['quality'])):
    df.loc[val_idx, 'fold'] = fold

if not os.path.exists("./output"):
    os.makedirs("./output")
df.to_csv("./output/train_stratified_folds_.csv")

# FOR A LARGE DATASET
# -------------------
# The process for creating the hold-out remains the same as stratified k-fold. For a
# dataset which has 1 million samples, we can create ten folds instead of 5 and keep one of those folds as hold-out.
# This means we will have 100k samples in the hold- out, and we will always calculate loss, accuracy and other
# metrics on this set and train on 900k samples.

#########################################################
###     STRATIFIED K FOLD FOR REGRESSION PROBLEMS     ###
#########################################################

# Mostly, simple k-fold cross-validation works for any regression problem. However, if you see that the distribution
# of targets is not consistent, you can use stratified k-fold.

# There are several choices for selecting the appropriate number of bins. If you have a lot of samples( > 10k,
# > 100k), then you don’t need to care about the number of bins. Just divide the data into 10 or 20 bins.
#
# If you do not have a lot of samples, you can use a simple rule like Sturge’s Rule to calculate the appropriate
# number of bins.

#Sturge’s rule:
#Number of Bins = 1 + log2(N)
number_obs_list = []
number_bins_list = []

for exponent in range(0, 7):
    number_obs = (10 ** exponent)
    number_bins = 1 + np.log2(number_obs)

    number_obs_list.append(number_obs)
    number_bins_list.append(number_bins)

plt.plot(number_bins_list, number_obs_list)
plt.xticks(range(1,int(max(number_bins_list)), 5))
#plt.yscale('log')
#plt.yticks(range(1,int(max(number_obs_list)), 500))
plt.xlabel("# of bins")
plt.ylabel("# of samples")
plt.ticklabel_format(style='plain')
plt.show()

#########################################################
######       Stratified K Fold Regression           #####
#########################################################

kfold = model_selection.StratifiedKFold(n_splits=5)

df['target'] = np.random.normal(50000, 23000, df.shape[0])

num_bins = 1 + np.log2(df.shape[0]) ##Sturge’s rule:

df['bins'] = pd.cut(df['target'], bins=int(num_bins), labels=False)
print(f'df bin values : {df.bins.value_counts()}')


for fold, (train_idx, val_idx) in enumerate(kfold.split(X=df, y=df.bins.values)):
    df.loc[val_idx, 'fold'] = fold

if not os.path.exists("./output"):
    os.makedirs("./output")
df.to_csv("./output/train_stratified_regression_folds.csv")
