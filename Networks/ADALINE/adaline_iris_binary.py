import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('URL:', data)

df = pd.read_csv(data,
                 header=None,
                 encoding='utf-8')

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-versicolor', 1, -1)
X = df.iloc[0:100, [0, 1, 2, 3]].values

X_train = np.concatenate([X[0:40], X[50:90]])
y_train = np.concatenate([y[0:40], y[50:90]])
X_test = np.concatenate([X[40:50], X[90:100]])
y_test = np.concatenate([y[40:50], y[90:100]])

from ADALINE import ADALINE

ad = ADALINE(epochs=10)
ad.fit(X_train, y_train)

predictions = ad.predict(X_test)

plt.plot(ad.cost_)
plt.xlabel('Epochs')
plt.ylabel('Loss (SSE)')
plt.title('Adaline - Learning rate 0.01')

plt.show()
