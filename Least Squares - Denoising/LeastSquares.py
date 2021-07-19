import numpy as np
import matplotlib.pyplot as plt

data = np.load('btc_price.npy')
plt.plot(data)
plt.show()
y = data.reshape(data.size, 1)
D = np.zeros(((data.size - 1), data.size))
for i in range(D.shape[0]):
    D[i][i] = 1
    D[i][i + 1] = -1


def denoise(D, y, lambdaa):
    x = np.linalg.inv(np.eye(D.shape[1], D.shape[1]) + (lambdaa * (D.T @ D))) @ y
    return x


plt.plot(denoise(D, y, 0))
plt.title('lambda = 0')
plt.show()
plt.plot(denoise(D, y, 10))
plt.title('lambda = 10')
plt.show()
plt.plot(denoise(D, y, 100))
plt.title('lambda = 100')
plt.show()
plt.plot(denoise(D, y, 1000))
plt.title('lambda = 1000')
plt.show()
plt.plot(denoise(D, y, 10000))
plt.title('lambda = 10000')
plt.show()
