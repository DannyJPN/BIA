

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def Schwefel(parameters):
    result=0
    for x in parameters:
        result += x * np.sin(np.sqrt(abs(x)))
    return 418.9829*len(parameters)-result

#for i in range(-500,500):
#    for j in range(-500,500):
#        parameters = (i,j)
#        print( Sphere(parameters))


X = np.arange(-500, 500, 1)
Y = np.arange(-500, 500, 1)
X, Y = np.meshgrid(X, Y)

Z = Schwefel((X,Y))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()
