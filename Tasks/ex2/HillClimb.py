import numpy as np
from matplotlib import pyplot as plt
import math
import random



from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import mpl_toolkits.mplot3d.axes3d as p3
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

#class containing the functions.
#Meant as Library class,outside python would be static,used in similar manner like C#'s Math lib or python's numpy lib
class BiaFunc:
    def __init__(self):
        pass
        
    def Griewangk(self,parameters):
        sumresult=0
        multiplyresult=1
        for i in range(len(parameters)):
            sumresult += (parameters[i]**2)/4000
        for i in range(len(parameters)):
            multiplyresult *= np.cos( parameters[i]/np.sqrt(i+1))
        
        return sumresult-multiplyresult+1
        
    def Levy(self,parameters):
        result=0
        wi=parameters[0]
        wd=parameters[len(parameters)-1]
        for i in range(len(parameters)-1):
            w = 1+(parameters[i]-1)/4
            result += ((w-1)**2) *(1+10*(np.sin(math.pi*w+1))**2)
        return (np.sin(math.pi*wi)**2)+result+((wd-1)**2)*(1+np.sin(2*math.pi*wd)**2)    
    
    
    def Michalewicz(self,parameters):
        result=0
        m=10
        for i in range(len(parameters)):
            result += np.sin(parameters[i])*(np.sin(((i+1)*(parameters[i]**2))/math.pi))**(2*m)
        return -result
    
    def Rastrigin(self,parameters):
        result=0
        d=len(parameters)
        for x in parameters:
            result += ((x**2)-10*np.cos(2*math.pi*x))
        return (10*d)+result
    
    def Rosenbrock(self,parameters):
        result=0
        for i in range(len(parameters)-1):
            result+=100*(parameters[i+1] - parameters[i]**2)**2 + (parameters[i]-1)**2
        return result
    
    def Schwefel(self,parameters):
        result=0
        for x in parameters:
            result += x * np.sin(np.sqrt(abs(x)))
        return 418.9829*len(parameters)-result
    
    def Sphere(self,parameters):
        result=0
        for x in parameters:
            result += x**2
        return result
    
    def Zakharov(self,parameters):
        first=0
        second=0
        third=0
        for i in range(len(parameters)):
            first += parameters[i]**2
            second += (0.5*(i+1)*parameters[i])**2
            third += (0.5*(i+1)*parameters[i])**4
        return first+second+third
    
    def Ackley(self,parameters):
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

#Solution class,encases the algorithm itself and helper functions
class Solution:
    def __init__(self, dimension,lowbound,upbound,generationsize,fitnessf,step):
        self.dimension=dimension
        self.lowbound=lowbound
        self.upbound=upbound
        self.generationsize=generationsize
        self.fitnessf=fitnessf
        self.step=step

#method creating a Generation of random numbers,neighbors of CENTER. Metropolis algorithm???
#produces list of n-tuples,where n = dimension -1
    def MakeGeneration(self,center,sigma):
        generation = []
        for i in range(self.generationsize):
            singlegen=[]
            for j in range(self.dimension-1):
                rannum=np.random.normal(center[j], sigma)
                singlegen.append(rannum)
            generation.append(singlegen)
        return generation
#method responsible for selecting the best member of given generation,based on its fitness.     
    def GetBestMember(self,generation):
        top = generation[0]
        for item in generation:
            if(self.fitnessf(item) < self.fitnessf(top)):
                top=item
        return top
            
#method producing Fitness specified by given fitness function                
    def GetFitness(self,elements):
        return self.fitnessf(elements)

#performs and displays Hill CLimbing algorithm 
    def HillClimb(self,iterations,generationrange):
#if the given dimension is 1 or lower,the algorithm will not occur,as it does not make sense
        if(self.dimension < 2):
            print("Invalid dimension");
            exit();
            
        #array initialization
        results = []
        indexes = []
        generationwinners = []
        gen_winners_args=[]
        #creating of the first initial solution
        totalwinner=[]
        for i in range(self.dimension-1):
            totalwinner.append(random.SystemRandom().uniform(self.lowbound,self.upbound))
    
        print("Totalwinner: "+str(totalwinner))
        #HC algorithm itself
        for i in range(iterations):
            indexes.append(i)
            #getting the generation
            generation = self.MakeGeneration(totalwinner,generationrange)
            #selecting the best member of generation
            winner = self.GetBestMember(generation)
            gen_winners_args.append(winner)
            generationwinners.append(self.GetFitness(winner))
            #evaluating if the best member will be accepted. if yes,returns the new total winner,else returns the former one
            totalwinner  = self.GetBestMember([totalwinner,winner])
            results.append(self.GetFitness(totalwinner))
            #debugging output. Writes out all the arrays after each iteration (iteration indexes,current temperature,winners of all generations,and all accepted results)
            print("Indexes: "+str(indexes))
            print("Winners: "+str(generationwinners))
            print("Results: "+str(results))
            
     #drawing out the algorithm in 2D
        if(self.dimension == 2):
            #plt.plot(indexes, generationwinners, label='Generation winners', linewidth='3')
            plt.plot(indexes, results, label='Hill Climbing', linewidth='1')
            
            plt.title('Hill CLimbing')
            plt.ylabel('Y axis')
            plt.xlabel('X axis')    
            
            plt.legend()
            plt.show()
#drawing out the algorithm in 3D.
        if(self.dimension == 3):
            #function which actually moves the result marker
            def UpdatePoint(n, x, y, z, point):                
                point.set_data(np.array([x[n], y[n]]))
                point.set_3d_properties(z[n], 'z')
                print("Iteration: "+str(n) + "\tPoint("+str(x[n])+ " , "+str(y[n])+ ")\twith fitness: "+str(z[n]))
                return point
            #plot initialization
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            
            
            
            X = np.arange(self.lowbound,self.upbound, self.step)
            Y = np.arange(self.lowbound,self.upbound, self.step)
            X, Y = np.meshgrid(X, Y)
            
            Z = self.fitnessf((X,Y))
            
             #ax = p3.Axes3D(fig)
            #ax.projection='3d'
            ax.set_xlim(self.lowbound,self.upbound)
            ax.set_ylim(self.lowbound,self.upbound)
            #ax.set_zlim(min(Z),max(Z))
            #transposing the array of winner points
            hillX=[]
            hillY=[]
            
            for duo in gen_winners_args:
                hillX.append(duo[0])
                hillY.append(duo[1])
                
            #defining the initial location of result marker
            point, = ax.plot([hillX[0]], [hillY[0]], [results[0]], 'o')
         
            # Plot the surface.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.seismic,linewidth=0, antialiased=True,alpha=0.5)
            ani=animation.FuncAnimation(fig, UpdatePoint, len(results), fargs=(hillX, hillY, results, point),interval=500,repeat_delay=5000)
            plt.show()

bfunc=BiaFunc()
sol=Solution(3,-5.12,5.12,2,bfunc.Sphere,0.1)

sol.HillClimb(200,0.01)
        

        
    
