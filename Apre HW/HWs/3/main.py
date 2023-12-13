import pandas as pd
import numpy as np

#*####################################################################################
#* 4)
#*####################################################################################

#* import data
from scipy.io.arff import loadarff
data = loadarff("kin8nm.arff")
df = pd.DataFrame(data[0])
num_columns = df.shape[1]

#* partition data
from sklearn.model_selection import train_test_split
X, y = df.iloc[:, 0:num_columns-1], df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.70, random_state=0)

#* linear regressor with Ridge regularization
from sklearn.linear_model import Ridge
rr = Ridge(alpha=0.1)

#* MLP_1 and MLP_2 regressors
from sklearn.neural_network import MLPRegressor
MLP1r = MLPRegressor(hidden_layer_sizes=(10, 10), activation="tanh", max_iter=500, random_state=0, early_stopping=True)
MLP2r = MLPRegressor(hidden_layer_sizes=(10, 10), activation="tanh", max_iter=500, random_state=0, early_stopping=False)

#* learn (".values" was added to avoid warnings)
rr.fit(X_train.values, y_train.values)
MLP1r.fit(X_train.values, y_train.values)
MLP2r.fit(X_train.values, y_train.values)

#* predict (".values" was added to avoid warnings)
rr_y_pred = rr.predict(X_test.values)
MLP1r_y_pred = MLP1r.predict(X_test.values)
MLP2r_y_pred = MLP2r.predict(X_test.values)

#* compute MAE
from sklearn.metrics import mean_absolute_error
y_true = y_test
rr_MAE = mean_absolute_error(y_true, rr_y_pred)
MLP1r_MAE = mean_absolute_error(y_true, MLP1r_y_pred)
MLP2r_MAE = mean_absolute_error(y_true, MLP2r_y_pred)
print("Ridge MAE: " + str(rr_MAE))
print("MLP1 MAE: " + str(MLP1r_MAE))
print("MLP2 MAE: " + str(MLP2r_MAE))

#*####################################################################################
#* 5)
#*####################################################################################

#* array of residuals (in absolute value)
rr_residuals = abs(y_true - rr_y_pred)
MLP1r_residuals = abs(y_true - MLP1r_y_pred)
MLP2r_residuals = abs(y_true - MLP2r_y_pred)

#* plot
import matplotlib.pyplot as plt
plt.boxplot(x=rr_residuals)
plt.title(label="Ridge")
plt.show()
plt.boxplot(x=MLP1r_residuals)
plt.title(label="MLP 1")
plt.show()
plt.boxplot(x=MLP2r_residuals)
plt.title(label="MLP 2")
plt.show()
plt.hist(x=rr_residuals)
plt.title(label="Ridge")
plt.show()
plt.hist(x=MLP1r_residuals)
plt.title(label="MLP 1")
plt.show()
plt.hist(x=MLP2r_residuals)
plt.title(label="MLP 2")
plt.show()

#*####################################################################################
#* 6)
#*####################################################################################

MLP1r_iterations = MLP1r.n_iter_
MLP2r_iterations = MLP2r.n_iter_
print('MLP1 Iterations: ' + str(MLP1r_iterations))
print('MLP2 Iterations: ' + str(MLP2r_iterations))
