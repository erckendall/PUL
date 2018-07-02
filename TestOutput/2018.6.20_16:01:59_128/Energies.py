import numpy as np
import matplotlib.pyplot as plt

total = np.load('egylist.npy')
potential = np.load('egpsilist.npy')
kinetic = np.load('ekandqlist.npy')

plt.plot(potential)
plt.plot(kinetic)
plt.plot(total)
plt.show()
