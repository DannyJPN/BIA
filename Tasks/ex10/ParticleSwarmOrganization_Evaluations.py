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
class Particle:
    funccount=0
    def __init__(self,lowbound,upbound,fitnessf,vmin,vmax,coors=[],velocity=[]):
        self.coors=coors
        self.velocity=velocity
        self.fitnessf=fitnessf
        self.vmin=vmin
        self.vmax=vmax
        self.lowbound=lowbound
        self.upbound=upbound
        if(len(self.velocity)<=0):
            for i in coors:
                self.velocity.append(0)
    def __str__(self):
        return "Coors: {0}\nVelocity: {1}\nFitness: {2}\n".format(self.coors,self.velocity,self.GetFitness())
        
    def GetFitness(self):
        Particle.funccount+=1
        return self.fitnessf(self.coors)
    def Relocate(self,c1,c2,pbest,gbest):
        np.random.seed(int(time.time()))
        newvel=list(np.zeros(len(self.velocity)))
        newcoors=list(np.zeros(len(self.coors)))
        newvel=self.AddVectors(newvel,self.velocity)
        firstdif = self.SubtractVectors(pbest.coors,self.coors)
        seconddif = self.SubtractVectors(gbest.coors,self.coors)
        
        r=np.random.uniform()
        firstmult=self.MultiplyVector(r*c1,firstdif)
        secondmult=self.MultiplyVector(r*c2,seconddif)
        
        
        newvel = self.AddVectors(newvel,firstmult)
        newvel = self.AddVectors(newvel,secondmult)
        newvel = self.FitInRange(newvel,self.vmin,self.vmax)
        newcoors=self.AddVectors(newvel,self.coors)
        newcoors=self.FitInRange(newcoors,self.lowbound,self.upbound)
        #print("R:{0}\nC1:{1}\nC2:{2}\nFDIF: {3}\nSDIF: {4}\nFMULT:{5}\nSMULT:{6}\nNewvel:{7}\nNewcoor:{8}\n".format(r,c1,c2,firstdif,seconddif,firstmult,secondmult,newvel,newcoors))

        self.coors=newcoors
        self.velocity=newvel
        
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
    def __init__(self, dimension,fitnessf,lowbound,upbound,vmin,vmax):
        self.dimension=dimension
        self.lowbound=lowbound
        self.upbound=upbound
        self.fitnessf=fitnessf
        self.vmin=vmin
        self.vmax=vmax

                
#method responsible for creating the initial swarm of random vectors within predefined boundaries
    def MakeSwarm(self,swarmsize):
        swarm = []
        for i in range(swarmsize):
            particlecoors=[]
            velocitycoors=[]
            for j in range(self.dimension):
                ranpart=np.random.uniform(self.lowbound, self.upbound)
                ranvel=np.random.uniform(self.vmin, self.vmax)
                velocitycoors.append(ranvel)
                particlecoors.append(ranpart)
            swarm.append(Particle(self.lowbound,self.upbound,self.fitnessf,self.vmin,self.vmax,particlecoors,velocitycoors))
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

#performs and displays PSO algorithm 
    def ParticleSwarmOptimization(self,swarmsize,maxfunccount,step,c1,c2):
#if the given dimension is lower than 1,the algorithm will not occur,as it does not make sense
        if(self.dimension < 1):
            print("Invalid dimension");
            exit();
        migrations=[]
        Index=[]
        swarm = self.MakeSwarm(swarmsize)
        migrations.append(copy.deepcopy(swarm))
        
        pbest = copy.deepcopy(swarm)
        gbest = self.GetBestMember(pbest)
        gbesthist=[]
        DEX=[]
        DEY=[]
        DEZ=[]
        cycle=0
        while(Particle.funccount < maxfunccount):
            #print("\nMIGRATION CYCLE {0} _________________\nGBEST:{1}\n".format(cycle,gbest.GetFitness()))
            #for mem in pbest:
                #print(mem.GetFitness())
            for i in range(len(swarm)):
                #print("Particle {0} relocated from {1} ".format(i,swarm[i].coors))
                swarm[i].Relocate(c1,c2,pbest[i],gbest)
                
                #print("to {0} ".format(swarm[i].coors))
                #print("Swarm : {0} vs PBEST: {1}".format(swarm[i].GetFitness(),pbest[i].GetFitness()))
                if(swarm[i].GetFitness() <= pbest[i].GetFitness()):
                    pbest[i] = swarm[i]
                    #print("PBEST REPLACED: {0}".format(pbest[i].GetFitness()))
                    #print("PBEST : {0} vs GBEST: {1}".format(pbest[i].GetFitness(),gbest.GetFitness()))
                    if(pbest[i].GetFitness() <= gbest.GetFitness()):
                        gbest=copy.deepcopy(pbest[i])
                        #print("GBEST REPLACED: {0}".format(gbest.GetFitness()))
            gbesthist.append(gbest)
            migrations.append(copy.deepcopy(pbest))
            #print("----------------------------")
            #for mem in pbest:
                #print(mem.GetFitness())
            #print("\nMIGRATION CYCLE {0} _________________\nGBEST:{1}\n".format(cycle,gbest.GetFitness()))        
            fitcount = Particle.funccount
            #print("\nMIGRATION CYCLE {0}\tWinner fitness {1}\tFunccount {2}".format(cycle,gbest.GetFitness(),fitcount))

            for mem in pbest: 
                Index.append(cycle)
                DEX.append(mem.coors[0])
                DEY.append(mem.coors[1])
                DEZ.append(mem.GetFitness())
                
            cycle +=1
        return self.GetBestMember(pbest)
        #dataf = pd.DataFrame({"swarm": Index ,"x" : DEX, "y" : DEY, "z" : DEZ})
        #for i in gbesthist:
            #print(i.GetFitness())



#main    

#sol=Solution(2,bfunc.Levy,-100,100,-20,20)
#sol.ParticleSwarmOptimization(20,10000,0.5,2,2)
#
bfunc=BiaFunc()

maxofe = 10000
popnum=30
dim=30
sols=[]

sols.append(Solution(dim,bfunc.Ackley,-50,50,-10,10))
sols.append(Solution(dim,bfunc.Rastrigin,-50,50,-10,10))
sols.append(Solution(dim,bfunc.Rosenbrock,-50,50,-10,10))
sols.append(Solution(dim,bfunc.Schwefel,-50,50,-10,10))
sols.append(Solution(dim,bfunc.Zakharov,-50,50,-10,10))
sols.append(Solution(dim,bfunc.Michalewicz,-50,50,-10,10))
sols.append(Solution(dim,bfunc.Levy,-50,50,-10,10))
sols.append(Solution(dim,bfunc.Griewangk,-50,50,-10,10))
sols.append(Solution(dim,bfunc.Sphere,-50,50,-10,10))  


expno=250
for sol in sols:
    winners=[]
    funcname=str(sol.fitnessf).split(" ")[2].split(".")[-1]
    f = open("PSO_{0}.csv".format(funcname), "w")
    f.write("Experiment;Best solution PSO\n")
    f.close()
    for i in range(expno):    
        winner = sol.ParticleSwarmOptimization(popnum,maxofe,0.5,1,1)
        winners.append(winner)
        print("Winner {0}({1}):\t{2}".format(i,funcname,winner.GetFitness()))
        f = open("PSO_{0}.csv".format(funcname), "a")
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
    f = open("PSO_{0}.csv".format(funcname), "a")
    f.write("Mean;{0}\n".format(mean))
    f.write("Stddev;{0}\n".format(stdev))
    
    f.close()


