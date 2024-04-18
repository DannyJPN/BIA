

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def Griewangk(parameters):
    sumresult=0
    multiplyresult=1
    for i in range(len(parameters)):
        sumresult += (parameters[i]**2)/4000
    for i in range(len(parameters)):
        multiplyresult *= np.cos( parameters[i]/np.sqrt(i+1))
    
    return sumresult-multiplyresult+1

#for i in range(-600,600):
#    for j in range(-600,600):
#        parameters = (i,j)
#        print( Griewangk(parameters))


X = np.arange(-600, 600, 1)
Y = np.arange(-600, 600, 1)
X, Y = np.meshgrid(X, Y)


Z = Griewangk((X,Y))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()
