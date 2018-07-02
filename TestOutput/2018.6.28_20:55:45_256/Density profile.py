# Note: Because we are working with a cubic grid, performing a spherical average is subject to some considerable limitations.
# Firstly, the total size of the grid in terms of resolution determines the total number of different directions which can be probed. i.e. a grid with 3 grid points
# in each of the three dimensions can only be probed in 13 directions through the centre. As the grid size increases, more directions can be probed,
# corresponding to more pairs of points which draw a line through the centre.
# (Though in a cubic grid there will be separate pairs of points lying along the same line in some directions in general)
# Because of the spacing, some directions will not be resolved as well as others. The three primary directions will be resolved best,
# and then the 45 degree angled lines will also be appreciably resolved in general. For other directions the resolution will get progressively worse.
# One could perform a spherical interpolation, to try to resolve the other directions better, but would this yield better results than an average from
# the 13 main directions? Maybe, but also much more difficult computationally.
# And the diagonals are not all equal in 3d - 3 different resolutions
# Also we require a centre, which is unnatural on a cubic grid with even-valued resol.


import numpy as np
import matplotlib.pyplot as plt

fc = np.load('plane_#400.npy')
resol = np.shape(fc)[0]
gridlength = 1.
trunc = 120
gridspace = float(gridlength/resol)
est_cen = np.unravel_index(np.argmax(fc), fc.shape)
max_den = fc[est_cen]
rge = min(est_cen)
sol = []



f = np.load('initial_f.npy')
dr = .00001
alpha = np.sqrt(max_den)
for i in np.arange(trunc):
    if (int(np.sqrt(alpha) * (i*gridspace / dr + 1))) < 900000:
        sol.append((alpha * f[int(np.sqrt(alpha) * (i*gridspace / dr + 1))])**2)
    else:
        sol.append(0)

north = []
east = []
south = []
west = []

for i in np.arange(rge):
    north.append(fc[est_cen[0],est_cen[1]+i])
    south.append(fc[est_cen[0],est_cen[1]-i])
    east.append(fc[est_cen[0]+i,est_cen[1]])
    west.append(fc[est_cen[0]-i,est_cen[1]])

n_trunc = []
s_trunc = []
e_trunc = []
w_trunc = []
avg = []
data = []
nfw = []

for i in np.arange(trunc):
    n_trunc.append(north[i])
    data.append(north[i])
    s_trunc.append(south[i])
    data.append(south[i])
    e_trunc.append(east[i])
    data.append(east[i])
    w_trunc.append(west[i])
    data.append(west[i])
    avg.append(np.average(data))
    data = []
    if i == 0:
        nfw.append(0)
    else:
        nfw.append(1/(i*gridspace)**3)

plt.loglog(n_trunc, label='N')
plt.loglog(s_trunc, label = 'S')
plt.loglog(e_trunc, label = 'E')
plt.loglog(w_trunc, label = 'W')
# plt.loglog(avg, label='Avg.')
plt.loglog(sol, label='soliton')
plt.loglog(nfw, label='1/r^3')
plt.legend()
plt.xlim(1,trunc-1)
plt.ylim(1,max_den)
plt.show()
