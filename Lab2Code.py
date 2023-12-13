import pandas as pd
import numpy as np

# *####################################################################################
# * 5)
# *####################################################################################

# * Import Data
from scipy.io.arff import loadarff

data = loadarff("pd_speech.arff")
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')
num_columns = df.shape[1]
df['class'] = pd.to_numeric(df["class"])

# * Pre-Process Data: scale data (relevant for kNN)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df_scaled = df.copy()
df_scaled.iloc[:, 0:num_columns - 1] = MinMaxScaler().fit_transform(df.iloc[:, 0:num_columns - 1])
df_scaled.iloc[:, 0:num_columns - 1] = StandardScaler().fit_transform(df.iloc[:, 0:num_columns - 1])

# * Partition Data
X, y = df_scaled.iloc[:, 0:num_columns - 1], df_scaled["class"]

# * Classifier
from sklearn.neighbors import KNeighborsClassifier

kNN_predictor = KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2, metric="minkowski")
from sklearn.naive_bayes import GaussianNB

NB_predictor = GaussianNB()

# * Cross-Validation and Cumulative Confusion Matrixes
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

folds = StratifiedKFold(n_splits=10)
kNN_cumulative = np.array([[0, 0], [0, 0]])
NB_cumulative = np.array([[0, 0], [0, 0]])
for train_k, test_k in folds.split(X, y):
    X_train, X_test = X.iloc[train_k], X.iloc[test_k]
    y_train, y_test = y.iloc[train_k], y.iloc[test_k]

    kNN_predictor.fit(X_train, y_train)
    y_pred = kNN_predictor.predict(X_test)
    cm = np.array(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    kNN_cumulative[0][0] += cm[0][0]
    kNN_cumulative[0][1] += cm[0][1]
    kNN_cumulative[1][0] += cm[1][0]
    kNN_cumulative[1][1] += cm[1][1]
for train_k, test_k in folds.split(X, y):
    X_train, X_test = X.iloc[train_k], X.iloc[test_k]
    y_train, y_test = y.iloc[train_k], y.iloc[test_k]

    NB_predictor.fit(X_train, y_train)
    y_pred = NB_predictor.predict(X_test)
    cm = np.array(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    NB_cumulative[0][0] += cm[0][0]
    NB_cumulative[0][1] += cm[0][1]
    NB_cumulative[1][0] += cm[1][0]
    NB_cumulative[1][1] += cm[1][1]
kNN_cumulative_confusion = pd.DataFrame(kNN_cumulative, index=['Healthy', 'Parkinsons'],
                                        columns=['Predicted Healthy', 'Predicted Parkinsons'])
NB_cumulative_confusion = pd.DataFrame(NB_cumulative, index=['Healthy', 'Parkinsons'],
                                       columns=['Predicted Healthy', 'Predicted Parkinsons'])

# * Plot
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(kNN_cumulative_confusion, annot=True, fmt='g')
plt.title(label="kNN")
plt.show()
sns.heatmap(NB_cumulative_confusion, annot=True, fmt='g')
plt.title(label="Naive Bayes")
plt.show()

# *####################################################################################
# * 6)
# *####################################################################################

# * Test Hypothesis
from sklearn.model_selection import cross_val_score

kNN_fold_accs = cross_val_score(kNN_predictor, X, y, cv=10, scoring="accuracy")
NB_fold_accs = cross_val_score(NB_predictor, X, y, cv=10, scoring="accuracy")
print(kNN_fold_accs)
print(NB_fold_accs)

from scipy import stats

# kNN is better than NB?
res = stats.ttest_rel(kNN_fold_accs, NB_fold_accs, alternative='greater')
print("kNN > NB?  pval=", res.pvalue)
# kNN is worse than NB?
res = stats.ttest_rel(kNN_fold_accs, NB_fold_accs, alternative='less')
print("kNN < NB?  pval=", res.pvalue)
# kNN is differs from NB?
res = stats.ttest_rel(kNN_fold_accs, NB_fold_accs, alternative='two-sided')
print("kNN != NB? pval=", res.pvalue)