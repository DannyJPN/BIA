import copy


import numpy as np
from matplotlib import pyplot as plt
import math
import random



from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import mpl_toolkits.mplot3d.axes3d as p3
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

#object of one city,basically only a named array of coordinations
class City:
    def __init__ (self):
        self.coors = []
    def __str__(self):
        return str(self.coors)
#generation member,in fact one path. its constructor takes the list of cities and performs a permutation,creates a random path
class Member:
    def __init__ (self,cities):
        self.trajectory=[]
        citylist=copy.deepcopy(cities)
        initlen=len(citylist)
        while(len(self.trajectory) < initlen):
            ind=int(random.SystemRandom().uniform(0,len(citylist)))
            if(citylist[ind] not in self.trajectory):
                self.trajectory.append(citylist[ind])
                citylist.pop(ind)
        
#method for mutating with given probability        
    def Mutate(self,prob):
        probability=random.SystemRandom().uniform(0,1)
        length = len(self.trajectory)
        if(probability < prob):
            a=int(random.SystemRandom().uniform(0,length))
            b=int(random.SystemRandom().uniform(0,length))
            tmp = self.trajectory[a]
            self.trajectory[a] = self.trajectory[b]
            self.trajectory[b] = tmp
    
    def __str__(self):
        res=""
        for i in self.trajectory:
            res=res+str(i)+"\n"
        res += + "Length: "+ str(self.GetFitness())
        return str(self.trajectory)
#calculating the fitness,the total trajectory length
    def GetFitness(self):
        suma=0
        for i in range(len(self.trajectory)-1):
            suma+=self.GetCityDistance(self.trajectory[i],self.trajectory[i+1])
         
        suma+=self.GetCityDistance(self.trajectory[-1],self.trajectory[0])
        return suma
#helping method,calculates Euclidean distance between two cities        
    def GetCityDistance(self,city1,city2):
        distsum = 0
        dimension = len(city1.coors)
        for i in range(dimension):
            distsum += (city2.coors[i] - (city1.coors[i]))**2
        
        return np.sqrt(distsum)
        
    
        
#class for solving the algorithm itself. Its constructor takes dimension of calculation and default set of cities
class Solution:
    def __init__(self,dimension,pointlist):
        self.pointlist =pointlist
        self.dimension = dimension
#method for generating random number different than toavoid        
    def RandomDifferent(self,toavoid,max):
        rannum=int(random.SystemRandom().uniform(0,max))
        while(rannum == toavoid):
           rannum=int(random.SystemRandom().uniform(0,max))
        return rannum
#generates members of asignle generation. Returns list of Members-generation        
    def GenerateMembers(self,memcount):
        gen = []
        for i in range(memcount):
            gen.append(Member(self.pointlist))
        return gen
#substitution for "in" operator. Can tell if a Member contains a specific City    
    def CityInMember(self,member,city):
        for item in member.trajectory:
            if(item.coors == city.coors):
                return True
        return False
#method for mating process. Produces a child of parA and parB,which are parent elements    
    def Mate(self,parA,parB):
        separindex = int(len(parA.trajectory)/2)
        #đprint("Separindex: "+str(separindex))
        offs=Member([])
        for i in range(separindex):
            offs.trajectory.append(parA.trajectory[i])
        
        i=0
        while(len(offs.trajectory) < len(parB.trajectory)):

            if(self.CityInMember(offs,parB.trajectory[i]) == False):
                offs.trajectory.append(parB.trajectory[i])
            i=i+1
            
        return offs
#determines the member of given generation with lowest fitness    
    def GetBestMember(self,memberlist):
        best = memberlist[0]
        for item in memberlist:
            if(best.GetFitness() > item.GetFitness()):
                best=item
        return best
#TravellingSalesmanAlgorithm itself.    
    def TSP(self,generationcount,memcount,mutationprobability=0.5):
        #if the given dimension is 1 or lower,the algorithm will not occur,as it does not make sense
        if(self.dimension < 2):
            print("Invalid dimension");
            exit();
        #initialization
        traces = []
        citycount = len(self.pointlist)
        generation = self.GenerateMembers(memcount)
        nextgeneration = copy.deepcopy(generation)
        
        for gen in range(generationcount):
            for mem in range(memcount):
                parentA = nextgeneration[mem]
                parentB = nextgeneration[self.RandomDifferent(mem,memcount)]
                offspring = self.Mate(parentA,parentB)
                offspring.Mutate(mutationprobability)
                
                if(offspring.GetFitness() < parentA.GetFitness()):
                    nextgeneration[mem] = offspring
            winner = self.GetBestMember(nextgeneration)
            traces.append(winner)
            generation=nextgeneration
            print([mem.GetFitness() for mem in nextgeneration])
                
            print("Generation "+str(gen))
            
            #drawing the paths
        if(self.dimension == 2):
            #function which actually changes the path
            def UpdatePath(n,points, traces):
                TSPX=[]
                TSPY=[]
                for city in traces[n].trajectory:
                    TSPX.append(city.coors[0])
                    TSPY.append(city.coors[1])
                TSPX.append(traces[n].trajectory[0].coors[0])
                TSPY.append(traces[n].trajectory[0].coors[1])
                print("Drawing "+str(n)+ ". generation with length "+ str(traces[n].GetFitness()))
                points.set_data(TSPX,TSPY)
                return points
            
            fig=plt.figure()
            TSPX=[]
            TSPY=[]
            for city in traces[0].trajectory:
                TSPX.append(city.coors[0])
                TSPY.append(city.coors[1])
            TSPX.append(traces[0].trajectory[0].coors[0])
            TSPY.append(traces[0].trajectory[0].coors[1])
            
            
            points, = plt.plot(TSPX, TSPY, '-o')  
            ani=animation.FuncAnimation(fig, UpdatePath, len(traces), fargs=( points,traces),interval=250,repeat_delay=10000,blit=False)
            plt.show()
            

class CityListGen:
    def __init__(self):
        pass
    
    def Generate(self,lowbound,upbound,dimension,citycount):
        self.cities=[]
        while(len(self.cities) < citycount):
            city=City()
            for i in range(dimension):
                city.coors.append(random.SystemRandom().uniform(lowbound,upbound))
            self.cities.append(city)
        return self.cities
                
                
                
                
cgen = CityListGen()
clist=cgen.Generate(-200,200,2,20)
for item in clist:
    print(str(item))
print("________________________________________________________")

sol = Solution(2,clist)
#generation=sol.GenerateMembers(7)
#for item in generation:
#    for c in item.trajectory:
#        print(str(c))
#    print(str(item.GetFitness()))
#    print("-----------")
sol.TSP(1000000,20)
                