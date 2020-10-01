'''
Gabriel Gibeau Sanchez - gibg2501
IFT799 - TP1
30 sept. 2020
'''

from sklearn.datasets import load_iris
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load iris dataset
iris_dataset = load_iris()


def get_inter_distances(class_data: list,
                        class_stats: list,
                        covar_data=None,
                        method='euclidian',
                        covar_method='classwise'):
    d = list()
    r = dict()

    # Get alll directional inter-classes distances with specified the norm
    if covar_method == 'overall' and covar_data is not None:
        # iv = np.linalg.inv(np.cov(iris.iloc[:, 0:4], rowvar=False))
        iv = np.linalg.inv(np.cov(covar_data, rowvar=False))

    for i in range(0, len(iris_dataset['target_names'])):
        remaining_classes = [*range(0, len(iris_dataset['target_names']))]
        remaining_classes.remove(i)
        if method == 'euclidian':
            for j in remaining_classes:
                d.clear()
                for _, row in class_data[i].iterrows():
                    d.append(distance.euclidean(row, class_stats[j].loc['mean']))
                r[iris_dataset['target_names'][j] + '-' + iris_dataset['target_names'][i]] = np.min(d)
        elif method == 'mahalanobis':
            if covar_method == 'classwise':
                iv = np.linalg.inv(np.cov(class_data[i], rowvar=False))
            for j in remaining_classes:
                d.clear()
                for _, row in class_data[i].iterrows():
                    d.append(distance.mahalanobis(row, class_stats[j].loc['mean'], iv))
                r[iris_dataset['target_names'][j] + '-' + iris_dataset['target_names'][i]] = np.min(d)

    return r


def get_intra_distances(class_data: pd.DataFrame,
                        class_stats: pd.Series,
                        covar_data=None,
                        method='euclidian',
                        covar_method='classwise'):
    d = list()
    r = dict()

    if covar_method == 'overall' and covar_data is not None:
        iv = np.linalg.inv(np.cov(covar_data, rowvar=False))

    for i in range(0, len(iris_dataset['target_names'])):
        numeric_data = class_data[i]
        d.clear()
        if method == 'euclidian':
            for _, row in numeric_data.iterrows():
                d.append(distance.euclidean(row, class_stats[i].loc['mean']))
        elif method == 'mahalanobis':
            if covar_method == 'classwise':
                iv = np.linalg.inv(np.cov(class_data[i], rowvar=False))
            for _, row in numeric_data.iterrows():
                d.append(distance.mahalanobis(row, class_stats[i].loc['mean'], iv))

        r[iris_dataset['target_names'][i]] = np.max(d)

    return r


# Put dataset in a panda Dataframe
targets = np.zeros((iris_dataset['data'].shape[0], 1))
targets[:, 0] = iris_dataset['target']

data = np.array(iris_dataset['data'])

cols = [fname for fname in iris_dataset['feature_names']]
cols.append('target')

iris = pd.DataFrame(np.concatenate((data, targets), axis=1), columns=cols)

for i in range(0, len(iris_dataset['target_names'])):
    iris.loc[iris['target'] == i, 'target'] = [iris_dataset['target_names'][i]] * np.sum(iris['target'] == i)

# Part 1
# Descriptive statistics
print(f'class-indpendent descriptives statistiques:\n{iris.describe()}')

# Here we get all descriptive stats for all classs and per-class, and then we compute the
# similtude analysis using mahalabobis, manhattan and euclidian distances
class_data = list()
class_stats = list()
intra_classs_distances = list()

for i in range(0, len(iris_dataset['target_names'])):
    unique_class = iris[iris['target'] == iris_dataset['target_names'][i]]

    class_data.append(unique_class)
    class_description = unique_class.describe()
    class_name = iris_dataset['target_names'][i]

    print(f'{class_name}:\n{class_description}')
    class_stats.append(class_description)

# get intra-class distances
nb_features_kept = 3
data_arg = [c.drop(columns=iris_dataset['feature_names'][nb_features_kept], axis=1) for c in class_data]
data_arg = [c.drop(columns='target', axis=1) for c in data_arg]
stats_arg = [c.drop(columns=iris_dataset['feature_names'][nb_features_kept], axis=1) for c in class_stats]
covar_data = iris.drop(columns=[iris_dataset['feature_names'][nb_features_kept], 'target'], axis=1)

intra_dist_euclid = get_intra_distances(data_arg, stats_arg)
intra_dist_mahalanobis = get_intra_distances(data_arg, stats_arg,
                                             covar_data=covar_data,
                                             method='mahalanobis')
intra_dist_mahalanobis_all = get_intra_distances(data_arg, stats_arg,
                                                 covar_data=covar_data,
                                                 method='mahalanobis',
                                                 covar_method='overall')

# get inter-class distances
inter_dist_euclid = get_inter_distances(data_arg, stats_arg)
inter_dist_mahalanobis = get_inter_distances(data_arg, stats_arg,
                                             covar_data=covar_data,
                                             method='mahalanobis')
inter_dist_mahalanobis_all = get_inter_distances(data_arg, stats_arg,
                                                 covar_data=covar_data,
                                                 method='mahalanobis',
                                                 covar_method='overall')

# verify if the classes are well separated
# if intra_dist_x < inter_dist_(y->x) => x & y are well separated
print('\nInter & intra-class distances using euclidian norm')
for i in range(0, len(iris_dataset['target_names'])):
    cname1 = iris_dataset['target_names'][i]
    for j in range(0, len(iris_dataset['target_names'])):
        cname2 = iris_dataset['target_names'][j]
        if cname1 != cname2:
            if inter_dist_euclid[f'{cname1}-{cname2}'] > intra_dist_euclid[cname2]:
                print(f'{cname1} and {cname2} are well separated.')
            else:
                print(f'{cname1} and {cname2} are overlaping!')

print('\nInter & intra-class distances using mahalanobis norm and classwise variance')
for i in range(0, len(iris_dataset['target_names'])):
    cname1 = iris_dataset['target_names'][i]
    for j in range(0, len(iris_dataset['target_names'])):
        cname2 = iris_dataset['target_names'][j]
        if cname1 != cname2:
            if inter_dist_mahalanobis[f'{cname1}-{cname2}'] > intra_dist_mahalanobis[cname2]:
                print(f'{cname1} and {cname2} are well separated.')
            else:
                print(f'{cname1} and {cname2} are overlaping!')

print('\nInter & intra-class distances using mahalanobis norm and overall covariance')
for i in range(0, len(iris_dataset['target_names'])):
    cname1 = iris_dataset['target_names'][i]
    for j in range(0, len(iris_dataset['target_names'])):
        cname2 = iris_dataset['target_names'][j]
        if cname1 != cname2:
            if inter_dist_mahalanobis[f'{cname1}-{cname2}'] > intra_dist_mahalanobis[cname2]:
                print(f'{cname1} and {cname2} are well separated.')
            else:
                print(f'{cname1} and {cname2} are overlaping!')

print('\n')

# Correlation between the features
print(iris.corr(method='pearson'))

# pairplot plots
# ax = sns.pairplot(iris, hue='target')
# ax._legend.remove()
# plt.legend(iris_dataset['target_names'], loc=4, bbox_to_anchor=(0.9, 0.0))
# plt.savefig('iris pairplot')
# plt.show()

# correlation plot
sns.heatmap(iris.corr(method='pearson'), cmap='coolwarm', annot=True)
plt.savefig('iris correlation')
plt.show()

scaler = StandardScaler()

# Normalize and scale the features
scaler = StandardScaler()
features = iris.iloc[:, 0:4]
features = scaler.fit_transform(features)
labels = iris['target']

iris_pca = PCA(n_components=3)
iris_pca.fit(features, labels)
iris_tf = iris_pca.transform(data)
pcomponents = pd.DataFrame(iris_tf)

for i in range(3):
    for j in range(3):
        if j > i:
            sns.scatterplot(pcomponents.iloc[:, i], pcomponents.iloc[:, j], hue=labels, cmap='Set1')
            plt.xlabel(f'compoosante principale {i}')
            plt.ylabel(f'compoosante principale {j}')
            plt.title(f'Nuages de points des \ncompoosantes principales {i} et {j}')
            plt.savefig(f'Nuages de points des \ncompoosantes principales {i} et {j}')
            plt.show()
