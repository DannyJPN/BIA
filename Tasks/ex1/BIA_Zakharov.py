from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def Zakharov(parameters):
    first=0
    second=0
    third=0
    for i in range(len(parameters)):
        first += parameters[i]**2
        second += (0.5*(i+1)*parameters[i])**2
        third += (0.5*(i+1)*parameters[i])**4
    return first+second+third

#for i in range(-32.768,32.768):
#    for j in range(-32.768,32.768):
#        parameters = (i,j)
#        print( Zakharov(parameters))


X = np.arange(-5,10, 0.5)
Y = np.arange(-5,10, 0.5)
X, Y = np.meshgrid(X, Y)

Z = Zakharov((X,Y))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()
