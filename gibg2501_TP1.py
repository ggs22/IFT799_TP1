from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt

# Load iris dataset
iris_dataset = load_iris()

# TODO: centroids & kmeans
# TODO: variance intra-classe
# TODO: cohesion & separation

# Put dataset in a panda Dataframe
targets = np.zeros((iris_dataset['data'].shape[0], 1))
targets[:, 0] = iris_dataset['target']

data = np.array(iris_dataset['data'])

cols = [fname for fname in iris_dataset['feature_names']]
cols.append('target')

iris = pd.DataFrame(np.concatenate((data, targets), axis=1), columns=cols)

for i in range(0, len(iris_dataset['target_names'])):
    iris.loc[iris['target'] == i,'target'] = [iris_dataset['target_names'][i]] * np.sum(iris['target'] == i)

# Descriptive statistics
print(iris.describe())

# Correlation between the features
print(iris.corr(method='pearson'))

# TODO: PCA -> plots -> decision frontiers

# pairplot plots
sns.pairplot(iris, hue='target')
plt.show()

# correlation plot
sns.heatmap(iris.corr(method='pearson'), cmap='coolwarm')
plt.show()

# sns.distplot(iris.iloc[:, 0:3], color=['1', '2', '3'])
# plt.show()

input("press any key to exit...")