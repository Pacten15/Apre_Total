import matplotlib.pyplot as plt
from sklearn import metrics, datasets, tree
from sklearn.model_selection import train_test_split

# 1. load and partition data
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# 2. learn classifier
predictor = tree.DecisionTreeClassifier(max_depth=3)
predictor.fit(X_train, y_train)

# 3. plot classifier
figure = plt.figure(figsize=(12, 6))
tree.plot_tree(predictor, feature_names=iris.feature_names, class_names=iris.target_names, impurity=False)
plt.show()

y_pred = predictor.predict(X_test)
print("accuracy on testing set:",  round(metrics.accuracy_score(y_test, y_pred),2))

single_pred = predictor.predict([X_test[0]])
print("prediction for instance",X_test[0],"is",iris.target_names[single_pred[0]])


import pandas as pd, numpy as np

# read titanic data
df = pd.read_csv('titanic.csv')

# remove non relevant variables
df.drop(["PassengerId","Name","Ticket","Cabin"], axis = 1, inplace=True)

# remove observations with missings
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)




