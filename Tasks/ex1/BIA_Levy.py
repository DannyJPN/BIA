

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def Levy(parameters):
    result=0
    wi=parameters[0]
    wd=parameters[len(parameters)-1]
    for i in range(len(parameters)-1):
        w = 1+(parameters[i]-1)/4
        result += ((w-1)**2) *(1+10*(np.sin(math.pi*w+1))**2)
    return (np.sin(math.pi*wi)**2)+result+((wd-1)**2)*(1+np.sin(2*math.pi*wd)**2)

#for i in range(-10,10):
#    for j in range(-10,10):
#        parameters = (i,j)
#        print( Levy(parameters))


X = np.arange(-10, 10, 0.5)
Y = np.arange(-10, 10, 0.5)
X, Y = np.meshgrid(X, Y)

Z = Levy((X,Y))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()
