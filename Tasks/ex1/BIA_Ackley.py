
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def Ackley(parameters):
    first=0
    second=0
    a=20
    b=0.2
    c=2*math.pi
    d=len(parameters)
    for x in parameters:
        first+=x**2
        second+=np.cos(c*x)
    return -a*np.exp(-b*np.sqrt(first/d)) - np.exp(second/d) +a + np.exp(1)

#for i in range(-32.768,32.768):
#    for j in range(-32.768,32.768):
#        parameters = (i,j)
#        print( Ackley(parameters))


X = np.arange(-32.768, 32.768, 0.5)
Y = np.arange(-32.768, 32.768, 0.5)
X, Y = np.meshgrid(X, Y)

Z = Ackley((X,Y))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()
