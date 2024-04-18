from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def Michalewicz(parameters):
    result=0
    m=10
    for i in range(len(parameters)):
        result += np.sin(parameters[i])*(np.sin(((i+1)*(parameters[i]**2))/math.pi))**(2*m)
    return -result

#for i in range(-32.768,32.768):
#    for j in range(-32.768,32.768):
#        parameters = (i,j)
#        print( Michalewicz(parameters))


X = np.arange(0, math.pi, 0.1)
Y = np.arange(0, math.pi, 0.1)
X, Y = np.meshgrid(X, Y)

Z = Michalewicz((X,Y))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()
