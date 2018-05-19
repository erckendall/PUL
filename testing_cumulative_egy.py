import numpy as np
import matplotlib.pyplot as plt

f = np.load('./TestOutput/30.0egpsi_cumulative.npy')

plt.plot(f)
plt.show()
