import matplotlib.pyplot as plt
from sklearn import metrics, datasets, tree
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_classif

#* Import data
data = loadarff("pd_speech.arff")
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')

#* Visualize data
print(df.head())
#print(df.describe())
#print(df.value_counts("class"))

#* Check nulls: there's none
#print(df.isnull().sum())

#* Check NAs: there's none
#df.dropna(inplace=True)
#df.reset_index(drop=True, inplace=True)

X, y = df.drop("class", axis=1), np.ravel(df['class'])

k_list = [5, 10, 40, 100, 250, 700]
acc_test_list = []
acc_train_list = []

for k in k_list:
    #* select features
    transformer = GenericUnivariateSelect(mutual_info_classif, mode='k_best', param=k)
    X_new = transformer.fit_transform(X, y)
    #* split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, train_size = 0.70, random_state=1)
    print("#training obs =", len(X_train), "\n#testing obs =", len(X_test))
    #* learn
    predictor = tree.DecisionTreeClassifier()
    predictor.fit(X_train, y_train)


    #* accurary
    y_pred_train = predictor.predict(X_train)
    y_pred_test  = predictor.predict(X_test)
    acc_train_list.append(metrics.accuracy_score(y_train, y_pred_train))
    acc_test_list.append(metrics.accuracy_score(y_test, y_pred_test))
    
#* final dataframe
data_acc  = []
for i in range(6):
    data_acc.append([k_list[i], acc_train_list[i], acc_test_list[i]])
    
df_final = pd.DataFrame(data_acc, columns = ['Features', 'Training Accuracy', 'Testing Accuracy'])

#* plot
plt.plot(k_list, acc_train_list, label = "Training Accuracy")
plt.plot(k_list, acc_test_list, label = "Testing Accuracy")
plt.legend()
plt.xlabel('x- Features')
plt.ylabel('y- Percentages')
plt.show()

#* Mostrar árvore
#figure = plt.figure(figsize=(12, 6))
#tree.plot_tree(predictor, impurity=False)
#plt.show()
