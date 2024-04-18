

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def Rosenbrock(parameters):
    result=0
    for i in range(len(parameters)-1):
        result+=100*(parameters[i+1] - parameters[i]**2)**2 + (parameters[i]-1)**2
    return result

#for i in range(-32.768,32.768):
#    for j in range(-32.768,32.768):
#        parameters = (i,j)
#        print( Rosenbrock(parameters))


X = np.arange(-5, 10, 0.1)
Y = np.arange(-5, 10, 0.1)
X, Y = np.meshgrid(X, Y)

Z = Rosenbrock((X,Y))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()
