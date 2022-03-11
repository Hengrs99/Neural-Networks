import pandas as pd
import numpy as np
from Adaline import AdalineGD
import matplotlib.pyplot as plt


data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('URL:', data)

df = pd.read_csv(data,
                 header=None,
                 encoding='utf-8')

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-versicolor', 1, -1)

X = df.iloc[0:100, [0, 2]].values

ad = AdalineGD(n_iter=10)
ad.fit(X, y)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax[0].plot(range(1, len(ad.cost_) + 1), np.log10(ad.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

plt.show()
