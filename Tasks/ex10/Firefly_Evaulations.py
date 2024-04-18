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
    funccount=0
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
        Firefly.funccount +=1
        return self.fitnessf(self.coors)
#calculates Light intensity for travelling towards given neighbor    
    def GetBrightnessForNeighbor(self,visibility,neighbor):
        initbright = self.GetFitness()
        distance =self.GetDistanceFromNeighbor(neighbor) 
        #print("{0} * {1}(e^{2})".format(initbright,np.exp((-1)*visibility * distance),(-1)*visibility * distance))
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
        #print("ATR {0}".format(attract))
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
        newfirefly = Firefly(self.lowbound,self.upbound,self.fitnessf,self.defattract,newcoors)
        if(newfirefly.GetFitness() <= self.GetFitness()):
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
    def FireflyOptimization(self,swarmsize,maxfunccount,step,alphacontrol,defattract,visibility,usevisibility=False):
#if the given dimension is lower than 1,the algorithm will not occur,as it does not make sense
        if(self.dimension < 1):
            print("Invalid dimension");
            exit();
        migrations=[]
        Index=[]
        swarm = self.MakeSwarm(swarmsize,defattract)
        migrations.append(copy.deepcopy(swarm))
        alphaflyidx = self.GetBestMemberIndex(swarm)
        #for i in range(len(swarm)):
            #print(str(swarm[i]))       
        DEX=[]
        DEY=[]
        DEZ=[]
        cycle=0
        while(Firefly.funccount < maxfunccount):
            
            
            for i in range(len(swarm)):
                #print("INNERSTART --------------------")
                markers=[]
                for j in range(len(swarm)):
                    
                    bj = swarm[j].GetBrightnessForNeighbor(visibility,swarm[i])
                    bi =swarm[i].GetBrightnessForNeighbor(visibility,swarm[j])
                    #print("Ij {0}/{1} Ii {2}/{3}".format(j,bj,i,bi))
                    if(bj < bi):
                        fireflyinit = copy.deepcopy(swarm[i])
                        fireflymarker= fireflyinit.GetLocationTowardsNeighbor(swarm[j],alphacontrol,visibility,usevisibility)
                        #print("Marker of {0}({1}) -> {2}({3}) has loc {4}".format(i,fireflyinit.coors,j,swarm[j].coors,fireflymarker.coors))
                        markers.append(copy.deepcopy(fireflymarker))
                #print("INNER END------------------")    
                if(len(markers) > 0):
                    swarm[i].coors= self.GetBestMember(markers).coors
                  
            
                #print("_____________________________")
                            
            #print("---------------------------------------------------------------")
            swarm[alphaflyidx].MoveRand(alphacontrol)
            migrations.append(copy.deepcopy(swarm))
            #for i in range(len(swarm)):
                #print(str(swarm[i]))
            fitcount = Firefly.funccount
            #print("\nMIGRATION CYCLE {0}\tWinner fitness {1}\tFunccount {2}".format(cycle,swarm[alphaflyidx].GetFitness(),fitcount))
            alphaflyidx = self.GetBestMemberIndex(swarm)





            for mem in swarm: 
                Index.append(cycle)
                DEX.append(mem.coors[0])
                DEY.append(mem.coors[1])
                DEZ.append(mem.GetFitness())
               
                
            cycle+=1
        return self.GetBestMember(swarm)
#main    
#bfunc=BiaFunc()
#sol=Solution(2,bfunc.Sphere,-100,100)
#
#sol.FireflyOptimization(5,50000,0.5,0.3,1,0.2,usevisibility=False)
bfunc=BiaFunc()

maxofe = 10000
popnum=30
dim=30
sols=[]

sols.append(Solution(dim,bfunc.Ackley,-50,50))
sols.append(Solution(dim,bfunc.Rastrigin,-50,50))
sols.append(Solution(dim,bfunc.Rosenbrock,-50,50))
sols.append(Solution(dim,bfunc.Schwefel,-50,50))
sols.append(Solution(dim,bfunc.Zakharov,-50,50))
sols.append(Solution(dim,bfunc.Michalewicz,-50,50))
sols.append(Solution(dim,bfunc.Levy,-50,50))
sols.append(Solution(dim,bfunc.Griewangk,-50,50))
sols.append(Solution(dim,bfunc.Sphere,-50,50))  

expno=250

for sol in sols:
    winners=[]
    funcname=str(sol.fitnessf).split(" ")[2].split(".")[-1]
    f = open("FA_{0}.csv".format(funcname), "w")
    f.write("Experiment;Best solution FA\n")
    f.close()
    for i in range(expno):    
        winner = sol.FireflyOptimization(popnum,maxofe,0.5,0.3,1,0.2,False)
        winners.append(winner)
        print("Winner {0}({1}):\t{2}".format(i,funcname,winner.GetFitness()))
        f = open("FA_{0}.csv".format(funcname), "a")
        f.write("{0};{1}\n".format(i+1,winner.GetFitness()))
        f.close()
    sumval=0
    for winner in winners:
        sumval+=winner.GetFitness()
    mean = sumval/len(winners)
    
    sumdist=0
    for winner in winners:
        sumdist += (winner.GetFitness() -mean)**2
    stdev = np.sqrt(sumdist/len(winners))
    f = open("FA_{0}.csv".format(funcname), "a")
    f.write("Mean;{0}\n".format(mean))
    f.write("Stddev;{0}\n".format(stdev))
    
    f.close()
