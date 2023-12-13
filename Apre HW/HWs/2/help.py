import pandas as pd, numpy as np
import matplotlib.pyplot as plt

#* Import Data
data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv', delimiter=',')
num_columns = data.shape[1]
print(data.dtypes)
#//print(data.value_counts("Outcome"))

#* Partition Data
X, y = data.iloc[:, 0:num_columns-1], data["Outcome"]

#* Classifier
from sklearn.neighbors import KNeighborsClassifier
predictor = KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="minkowski")

#* Cross-Validation and Confunsion Matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
folds = StratifiedKFold(n_splits=10)
cms = []
for train_k, test_k in folds.split(X, y):
    X_train, X_test = X.iloc[train_k], X.iloc[test_k]
    y_train, y_test = y.iloc[train_k], y.iloc[test_k]

    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)
    cm = np.array(confusion_matrix(y_test, y_pred, labels=[0,1]))
    confusion = pd.DataFrame(cm, index=['Diabetic', 'Healthy'], columns=['Predicted Diabetes', 'Predicted Healthy'])
    print(confusion)
    break

