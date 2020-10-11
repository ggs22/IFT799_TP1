import numpy as np
import sklearn as sk
import sklearn.metrics


from scipy.spatial.distance import jaccard

arr1 = np.array([0, 1, 1, 0, 1, 0, 1])
arr2 = np.array([1, 1, 1, 0, 1, 0, 1])

print(f'Correlation coeff (Pearson\'s method):\n{np.corrcoef(arr1, arr2)}')

print(f'Jaccard-Needham: {jaccard(arr1, arr2)}')
print(f'Jaccard: {sk.metrics.jaccard_score(arr1, arr2)}')
# print(f'Jaccard: {sk.metrics.jaccard_similarity_score(arr1, arr2)}')

arr1 = np.array([0, 1, 1])
arr2 = np.array([1, 1, 1])

print(f'Jaccard-Needham: {jaccard(arr1, arr2)}')
print(f'Jaccard: {sk.metrics.jaccard_score(arr1, arr2)}')
# print(f'Jaccard: {sk.metrics.jaccard_similarity_score(arr1, arr2)}')

arr1 = np.array([1, 1, 1])
arr2 = np.array([0, 0, 0])

print(f'Jaccard-Needham: {jaccard(arr1, arr2)}')
print(f'Jaccard: {sk.metrics.jaccard_score(arr1, arr2)}')
# print(f'Jaccard: {sk.metrics.jaccard_similarity_score(arr1, arr2)}')

arr1 = np.array([1, 0, 0])
arr2 = np.array([0, 0, 0])

print(f'Jaccard-Needham: {jaccard(arr1, arr2)}')
print(f'Jaccard: {sk.metrics.jaccard_score(arr1, arr2)}')
# print(f'Jaccard: {sk.metrics.jaccard_similarity_score(arr1, arr2)}')

'''
Rule associations
'''
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# df = pd.read_excel('Online_Retail.xlsx')
df = pd.read_csv('retail_data_csv.csv', index_col=0)
df.head()

'''
some data pre-processing...
'''
# strip trailing & leading white spaces in descriptions
df['Description'] = df['Description'].str.strip()

# drop entries without invoice number
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)

# remove credit cards transactions
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

'''
Arbitrarily keep inly Franche data
'''

basket = (df[df['Country'] == 'France']
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


def encode_units(x):
    return int(x >= 1)

basket_set = basket.applymap(encode_units)
basket_set.drop('POSTAGE', inplace=True, axis=1)

frequent_itemsets = apriori(basket_set, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)


df_q4 = pd.DataFrame(data=np.array([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                                    [1, 24, 12, 31, 15, 22, 29, 40, 33, 38],
                                    ['a d e f',
                                     'a b c e',
                                     'a b d e',
                                     'a c e',
                                     'b c e f',
                                     'b d e',
                                     'c d e',
                                     'a b c f',
                                     'a d e',
                                     'a b c e']]).T,
                     columns=['CustomerID', 'TransactionID', 'Items'])

df_q4['CustomerID'] = df_q4['CustomerID'].astype(int)
df_q4['TransactionID'] = df_q4['TransactionID'].astype(int)

basket_q4 = df_q4.groupby(['CustomerID', 'Items'])['TransactionID'].sum().unstack().fillna(0)

basket_q4 = basket_q4.applymap(encode_units)

frequent_itemsets_q4 = apriori(basket_q4, min_support=0.05, use_colnames=True)
rules_q4 = association_rules(frequent_itemsets_q4, metric='lift', min_threshold=1)

dataset = [['a', 'c', 'd', 'e'],
           ['a', 'b', 'c', 'e'],
           ['a', 'b', 'd', 'e'],
           ['c', 'd', 'e'],
           ['a', 'b', 'c', 'd', 'e'],
           ['a', 'c', 'e']]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)


df_q4_2 = pd.DataFrame(te_ary, columns=te.columns_)

# df_q4_2['CustomerID'] = df_q4_2['CustomerID'].astype(int)
# df_q4_2['TransactionID'] = df_q4_2['TransactionID'].astype(int)
#
# basket_q4_2 = df_q4_2.groupby(['CustomerID']).all().loc[:, 'a':'f']
#
# basket_q4_2 = basket_q4_2.applymap(encode_units)

frequent_itemsets_q4_2 = apriori(df_q4_2, min_support=0.01, use_colnames=True)
rules_q4_2 = association_rules(frequent_itemsets_q4_2, metric='lift', min_threshold=1)

'''
For a priori algorithm example see:
https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
'''

import matplotlib.pyplot as plt

x = np.array([1, 0, 0.5, 1.5])
y = np.array([0, 1, 1.5, 0.5])

plt.scatter(x,y)

input('Press any key to exit...')