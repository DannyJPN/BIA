import numpy as np
from matplotlib import pyplot as plt
import math
import random
import copy
import pandas as pd
import time
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
class Firefly:
    def __init__(self,lowbound,upbound,fitnessf,defattract,coors=[]):
        self.coors=coors
        self.fitnessf=fitnessf
        self.defattract = defattract
        self.lowbound=lowbound
        self.upbound=upbound
        
    def __str__(self):
        return "Coors: {0}\nFitness: {1}\n".format(self.coors,self.GetFitness())
#calculates distance of this firefly from given neighbor    
    def GetDistanceFromNeighbor(self,neighbor):
        distsum = 0
        dimension = len(self.coors)
        for i in range(dimension):
            distsum += (neighbor.coors[i] - (self.coors[i]))**2
        
        return np.sqrt(distsum)    
        
    def GetFitness(self):
        return self.fitnessf(self.coors)
#calculates Light intensity for travelling towards given neighbor    
    def GetBrightnessForNeighbor(self,visibility,neighbor):
        initbright = self.GetFitness()
        distance =self.GetDistanceFromNeighbor(neighbor) 
        return initbright * np.exp((-1)*visibility * distance)
#calculates attractiveness of a firefly towards    
    def GetAttractivenessForNeighbor(self,neighbor,visibility,usevisibility=False):
        distance =self.GetDistanceFromNeighbor(neighbor)
        attract = 0
        if(usevisibility):
            attract = self.defattract * np.exp((-1)*visibility * (distance**2))
        else:
            attract = self.defattract/(1+distance)
        return attract
#method which generates position marker of a firefly towards given neighbor        
    def GetLocationTowardsNeighbor(self,neighbor,alphacontrol,visibility,usevisibility=False):
        gausvect = self.GetRandomGaussianVector()
        attract = self.GetAttractivenessForNeighbor(neighbor,visibility,usevisibility)
        subtract = self.SubtractVectors(neighbor.coors,self.coors)
        middlemember = self.MultiplyVector(attract,subtract)
        newcoors = self.AddVectors(self.coors,middlemember)
        randvect = self.MultiplyVector(alphacontrol,gausvect)
        newcoors = self.FitInRange(self.AddVectors(newcoors,randvect),self.lowbound,self.upbound)
        return Firefly(self.lowbound,self.upbound,self.fitnessf,self.defattract,newcoors)
#method for random movement of alpha firefly        
    def MoveRand(self,alphacontrol):
        gausvect = self.GetRandomGaussianVector()
        randvect = self.MultiplyVector(alphacontrol,gausvect)
        newcoors = self.AddVectors(self.coors,randvect)
        if(self.fitnessf(newcoors) <= self.GetFitness()):
            self.coors = self.FitInRange(newcoors,self.lowbound,self.upbound)
        
    def GetRandomGaussianVector(self):
        gausvect=[]
        for coor in self.coors:
            gausvect.append(np.random.normal(0,1))
        return gausvect
#method for checking if the vector is within borders 
    def FitInRange(self,vector,low,up):
        refined=[]
        for i in vector:
            if(i < low):
                refined.append(low)
            elif(i> up):
                refined.append(up)
            else:
                refined.append(i)
        return refined
#method for adding two vectors        
    def AddVectors(self,x,y):
        result = []
        if(len(x) != len(y)):
            print("Lengths inequal: {0} vs {1}".format(len(x) , len(y)))
            return result
        for i in range(len(x)):
            result.append(x[i]+y[i])
        return result
#method for subtracting two vectors               
    def SubtractVectors(self,x,y):
        result = []
        if(len(x) != len(y)):
            print("Lengths inequal: {0} vs {1}".format(len(x) , len(y)))
            return result
        for i in range(len(x)):
            result.append(x[i]-y[i])
        return result
#method for multiplying a vector with a scalar
    def MultiplyVector(self,A,x):
        result = []
        for i in range(len(x)):
            result.append(A*x[i])
        return result

class Solution:
    def __init__(self, dimension,fitnessf,lowbound,upbound):
        self.dimension=dimension
        self.lowbound=lowbound
        self.upbound=upbound
        self.fitnessf=fitnessf


                
#method responsible for creating the initial swarm of random vectors within predefined boundaries
    def MakeSwarm(self,swarmsize,defattract=1):
        swarm = []
        for i in range(swarmsize):
            fireflycoors=[]
            for j in range(self.dimension):
                ranpart=np.random.uniform(self.lowbound, self.upbound)
                fireflycoors.append(ranpart)
            swarm.append(Firefly(self.lowbound,self.upbound,self.fitnessf,defattract,fireflycoors))
        return swarm
    

#method responsible for selecting the best member of given swarm,based on its fitness.     
    def GetBestMember(self,swarm):
        top = swarm[0]
        for item in swarm:
            if(item.GetFitness() <= top.GetFitness()):
                top=item
        return top
#method responsible for selecting the index of best member of given swarm,based on its fitness,except the indexes specified in toavoid     
    def GetBestMemberIndex(self,swarm,toavoid=[]):
        top = 0
        for i in range(len(swarm)):
            if(swarm[i].GetFitness() <= swarm[top].GetFitness() and i not in toavoid):
                top=i
        return top
            

    
    #method for generating random number except those in toavoid        
    def RandomDifferent(self,max,toavoid=[]):
        rannum=int(random.SystemRandom().uniform(0,max))
        while(rannum in toavoid):
           rannum=int(random.SystemRandom().uniform(0,max))
        return rannum

#performs and displays FA algorithm 
    def FireflyOptimization(self,swarmsize,migrationrounds,step,alphacontrol,defattract,visibility,usevisibility=False):
#if the given dimension is lower than 1,the algorithm will not occur,as it does not make sense
        if(self.dimension < 1):
            print("Invalid dimension");
            exit();
        migrations=[]
        Index=[]
        swarm = self.MakeSwarm(swarmsize,defattract)
        migrations.append(copy.deepcopy(swarm))
        alphaflyidx = self.GetBestMemberIndex(swarm)
        for i in range(len(swarm)):
            print(str(swarm[i]))       
        DEX=[]
        DEY=[]
        DEZ=[]
        for cycle in range(migrationrounds):
            print("\nMIGRATION CYCLE {0} _________________\nWinner fitness {1}".format(cycle,swarm[alphaflyidx].GetFitness()))
            
            for i in range(len(swarm)):
                
                for j in range(len(swarm)):
                
                    if(swarm[j].GetBrightnessForNeighbor(visibility,swarm[i]) < swarm[i].GetBrightnessForNeighbor(visibility,swarm[j])):
                        fireflyinit = copy.deepcopy(swarm[i])
                        fireflymarker= fireflyinit.GetLocationTowardsNeighbor(swarm[j],alphacontrol,visibility,usevisibility)
                        print("Marker of {0}({1}) for moving towards {2}({3}) has loc \n{4}".format(i,fireflyinit.GetFitness(),j,swarm[j].GetFitness(),str(fireflymarker)))
                        if(fireflymarker.GetFitness() <= swarm[i].GetFitness()):
                            swarm[i]=fireflymarker
            

                            
            print("---------------------------------------------------------------")
            swarm[alphaflyidx].MoveRand(alphacontrol)
            migrations.append(copy.deepcopy(swarm))
            for i in range(len(swarm)):
                print(str(swarm[i]))
            
            
            alphaflyidx = self.GetBestMemberIndex(swarm)





            for mem in swarm: 
                Index.append(cycle)
                DEX.append(mem.coors[0])
                DEY.append(mem.coors[1])
                DEZ.append(mem.GetFitness())
               
                
            
        dataf = pd.DataFrame({"swarm": Index ,"x" : DEX, "y" : DEY, "z" : DEZ})
       



#drawing out the algorithm in 3D.
        
        if(self.dimension == 2):
        #function which actually moves the result markers
        

            def UpdatePoint(n,df, points):
                data=df[df['swarm']==n]
                points.set_data(data.x ,data.y)
                points.set_3d_properties(data.z)
                #print("swarm: "+str(list(data.swarm)) + "\tPoint("+str(list(data.x))+ " , "+str(list(data.y))+ ")\twith fitness: "+str(list(data.z)))
                print("swarm: {0}".format(data.swarm))
                print("X:\n {0}".format(data.x))
                print("Y:\n {0}".format(data.y))
                print("Z:\n {0}".format(data.z))
                print("__________________________________")
                
                return points,
            #plot initialization
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            X = np.arange(self.lowbound,self.upbound, step)
            Y = np.arange(self.lowbound,self.upbound, step)
            X, Y = np.meshgrid(X, Y)
            Z = self.fitnessf((X,Y))
            
            
            #ax = p3.Axes3D(fig)
            #ax.projection='3d'
            ax.set_xlim(self.lowbound,self.upbound)
            ax.set_ylim(self.lowbound,self.upbound)
            #ax.set_zlim(min(Z),max(Z))
           
            
          
            
            
            data=dataf[dataf['swarm']==0]
            #defining the initial location of result marker
            points, = ax.plot(data.x, data.y, data.z, marker='o',linestyle="")
            # Plot the surface.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=True,alpha=0.5)
            ani=animation.FuncAnimation(fig, UpdatePoint, len(migrations), fargs=(dataf, points),interval=500,repeat_delay=5000)
            
            
            plt.show()

#main    
bfunc=BiaFunc()
sol=Solution(2,bfunc.Ackley,-1000,1000)

sol.FireflyOptimization(30,50,0.5,0.5,1,20,usevisibility=False)
#swarm = sol.MakeSwarm(4)
#newswarm=copy.deepcopy(swarm)
#visibility = 20
#alphacontrol = 0.3
#usevisibility=False
#for s in swarm:
#    print(str(s))
#print("------------------------------------------")
#alphaflyidx =  sol.GetBestMemberIndex(swarm)
#           
#                
#for i in range(len(swarm)):
#    
#    for j in range(len(swarm)):
#       
#        if(swarm[j].GetBrightnessForNeighbor(visibility,swarm[i]) < swarm[i].GetBrightnessForNeighbor(visibility,swarm[j])):
#            fireflyinit = copy.deepcopy(swarm[i])
#            fireflymarker= fireflyinit.GetLocationTowardsNeighbor(swarm[j],alphacontrol,visibility,usevisibility)
#            print("Marker of {0}({1}) for moving towards {2}({3}) has loc \n{4}".format(i,fireflyinit.GetFitness(),j,swarm[j].GetFitness(),str(fireflymarker)))
#            if(fireflymarker.GetFitness() <= swarm[i].GetFitness()):
#                swarm[i]=fireflymarker
#
#print("------------------------------------------")
#for s in swarm:
#    print(str(s))
#