import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('URL:', s)

df = pd.read_csv(s,
                 header=None,
                 encoding='utf-8')

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-versicolor', 1, -1)
X = df.iloc[0:100, [0, 1, 2, 3]].values

X_train = np.concatenate([X[0:40], X[50:90]])
y_train = np.concatenate([y[0:40], y[50:90]])
X_test = np.concatenate([X[40:50], X[90:100]])
y_test = np.concatenate([y[40:50], y[90:100]])

plt.scatter(X[:50, 0], X[:50, 2], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 2], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

plt.scatter(X[:50, 1], X[:50, 3], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 1], X[50:100, 3], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal width [cm)')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

from Perceptron import Perceptron

ppn = Perceptron(eta=0.1, epochs=10)
ppn.fit(X_train, y_train)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

predicted_data = ppn.predict(X_test)
errors = 0
for pred, gt in zip(predicted_data, y_test):
    if pred != gt:
        errors += 1
