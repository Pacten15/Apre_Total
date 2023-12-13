import pandas as pd

# Reading the CSV file
df = pd.read_csv("Iris.csv")

# Printing top 5 rows
df.head()

df = pd.read_json("iris.json")
df.head()

from scipy.io.arff import loadarff

# Reading the ARFF file
data = loadarff('iris_arff.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')
df.head()
df.shape
# df.info()
df.describe()
df.value_counts("class")

import numpy as np

df.loc[0:1,'sepalwidth'] = np.nan
df.head()
df.isnull().sum()
df.dropna().head()

from sklearn.impute import SimpleImputer

# imputation strategies: mean, median, most_frequent
imp = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
df[['sepalwidth']] = imp.fit_transform(df[['sepalwidth']])
df.head()

unique_sepal = df.drop_duplicates(subset=["sepallength","sepalwidth"])
unique_sepal.shape


import matplotlib.pyplot as plt
import seaborn as sns


sns.countplot(x='class', data=df)
plt.show()


sns.scatterplot(x='sepallength', y='sepalwidth', hue='class', data=df)

# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

sns.scatterplot(x='petallength', y='petalwidth', hue='class', data=df)
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

plot = sns.FacetGrid(df, hue="class")
plot.map(sns.histplot, "sepallength").add_legend()

plot = sns.FacetGrid(df, hue="class")
plot.map(sns.histplot, "sepalwidth").add_legend()

plt.show()

sns.boxplot(x="class", y='sepallength', data=df)
plt.show()
sns.boxplot(x="class", y='sepalwidth', data=df)
plt.show()


sns.pairplot(df, hue='class', height=2)
plt.show()

print(df.corr(method='pearson'))

sns.heatmap(df.corr(method='pearson'), annot=True)
plt.show()

sns.heatmap(df.corr(method='pearson'), annot=True)
plt.show()


X = df.drop('class', axis=1)
y = df['class']

from sklearn.feature_selection import f_classif

fimportance = f_classif(X, y)

print('features', X.columns.values)
print('scores', fimportance[0])
print('pvalues', fimportance[1])