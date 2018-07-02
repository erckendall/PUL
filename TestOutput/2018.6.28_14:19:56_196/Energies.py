import numpy as np
import matplotlib.pyplot as plt

total196 = np.load('egylist.npy')
total256 = np.load('egylist 256.npy')
total288 = np.load('egylist 288.npy')


plt.plot(total196, label='196 sf 5')
plt.plot(total256, label='256 sf 5')
# plt.xlim(0,100)
plt.plot(total288, label='288 sf 10')
plt.legend()
plt.show()
