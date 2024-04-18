

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def Rastrigin(parameters):
    result=0
    d=len(parameters)
    for x in parameters:
        result += ((x**2)-10*np.cos(2*math.pi*x))
    return (10*d)+result

#for i in range(-5.12,5.12):
#    for j in range(-5.12,5.12):
#        parameters = (i,j)
#        print( Rastrigin(parameters))


X = np.arange(-5.12, 5.12, 0.1)
Y = np.arange(-5.12, 5.12, 0.1)
X, Y = np.meshgrid(X, Y)

Z = Rastrigin((X,Y))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()
