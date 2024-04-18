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

#object of one node,basically only a named array of coordinations
class Node:
    def __init__ (self,index):
        self.index=index
        self.coors = []
    def __str__(self):
        return str("{0}:{1}".format(self.index,self.coors))
#generation member,in fact one path. its constructor takes the list of cities and performs a permutation,creates a random path
class Ant:
    def __init__ (self,id):
        self.trajectory=[]
        self.id=id
        

    def __str__(self):
        res="Ant {0}\t".format(self.id)
        for i in self.trajectory:
            res=res+str(i.index)+","
        res +=  "\tLength: " + str(self.GetFitness())
        return res
        
        
#calculating the fitness,the total trajectory length
    def GetFitness(self):
        suma=0
        if (len(self.trajectory) <=0):
            return 0
        for i in range(len(self.trajectory)-1):
            suma+=self.GetNodeDistance(self.trajectory[i],self.trajectory[i+1])
         
        suma+=self.GetNodeDistance(self.trajectory[-1],self.trajectory[0])
        return suma
#helping method,calculates Euclidean distance between two cities        
    def GetNodeDistance(self,node1,node2):
        distsum = 0
        dimension = len(node1.coors)
        for i in range(dimension):
            distsum += (node2.coors[i] - (node1.coors[i]))**2
        
        return np.sqrt(distsum)
#substitution for "in" operator. Can tell if a Member contains a specific Node    
    def NodeTraversed(self,node):
        for item in self.trajectory:
            if(item.coors == node.coors  and item.index == node.index):
                return True
        return False        


 
        
    def FindPath(self,Alpha,Beta,nodelist,pheromatrix,visimatrix):
        #print("ANT {0} START____________________________________".format(self.id))
        #print("Ant {0} starting in node {1}".format(self.id,self.trajectory[-1].index))
        while(len(self.trajectory) < len(nodelist)):
            
            bottom=0
            probabilities={}    
            for node in nodelist:
                if (self.NodeTraversed(node) == False):
                    bottom+=pheromatrix[self.trajectory[-1].index][node.index]**Alpha * visimatrix[self.trajectory[-1].index][node.index]**Beta
            for node in nodelist:
                if (self.NodeTraversed(node) == False):
                    top=pheromatrix[self.trajectory[-1].index][node.index]**Alpha * visimatrix[self.trajectory[-1].index][node.index]**Beta
                    probabilities[node] = top/bottom
            rannum = np.random.uniform(0,1)
            #cumulative_probability=probabilities[list(probabilities.keys())[0]]
            cumulative_probability=0
            #print("Rannum {0}".format(rannum))
            #for p in probabilities.keys():
                #print("ID:{0}\tProbab:{1}".format(p.index,probabilities[p]))
            #print("SELECTNODE________________")
            for node in probabilities.keys():
                cumulative_probability+=probabilities[node]
                #print("Rannum {0} vs Cumulprobab {1}".format(rannum,cumulative_probability))
               
                if(rannum<cumulative_probability):
                    self.trajectory.append(node)
                    #print("Ant {0} visits node {1}".format(self.id,node.index))
                    break
                
                    
            #print("SELECTEDNODE________________")
        #print("ANT {0} FINISH____________________________________".format(self.id))

            
#class for solving the algorithm itself. Its constructor takes dimension of calculation and default set of cities
class Solution:
    def __init__(self,dimension,nodelist):
        self.nodelist =nodelist
        self.dimension = dimension
        self.visimatrix = None
        self.pheromatrix=None
#method for generating random number different than toavoid        
    def RandomDifferent(self,toavoid,max):
        rannum=int(random.SystemRandom().uniform(0,max))
        while(rannum == toavoid):
           rannum=int(random.SystemRandom().uniform(0,max))
        return rannum
#generates members of asignle generation. Returns list of Members-generation        
    def GenerateAntColony(self,colonysize):
        gen = []
        for i in range(colonysize):
            gen.append(Ant(i))
        return gen

#method for calculating distance between two given nodes
    def GetNodeVisibility(self,node1,node2):
        distsum = 0
        dimension = len(node1.coors)
        for i in range(self.dimension):
            distsum += (node2.coors[i] - (node1.coors[i]))**2
        res=np.sqrt(distsum)
        if(res == 0):
            return 0
        return np.sqrt(distsum)**(-1)
#distance matrix generation  
    def GetVisibilityMatrix(self):
        matrix=[]
        for node1 in self.nodelist:
            matrix.append([])
            for node2 in self.nodelist:
                matrix[-1].append(self.GetNodeVisibility(node1,node2))
        return matrix
#pheromone matrix generation        
    def GetPheromoneMatrix(self):
        matrix=[]
        for node1 in self.nodelist:
            matrix.append([])
            for node2 in self.nodelist:
                matrix[-1].append(1)
        return matrix      
       
#determines the member of given generation with lowest fitness    
    def GetBestAnt(self,colony):
        best = colony[0]
        for item in colony:
            if(best.GetFitness() > item.GetFitness()):
                best=item
        return best
        
        
    def UpdatePheromons(self,colony,evap_coef,Q):
        for i in range(len(self.pheromatrix)):
            for j in range(len(self.pheromatrix)):
                self.pheromatrix[i][j] *=evap_coef
                
        for ant in colony:
            for i in range(len(ant.trajectory)-1):
                self.pheromatrix[ant.trajectory[i].index][ant.trajectory[i+1].index] += Q/ant.GetFitness()
            self.pheromatrix[ant.trajectory[-1].index][ant.trajectory[0].index] += Q/ant.GetFitness()
            #print("Pheromatrix influenced by ant {0} and his power {1}".format(ant.id,Q/ant.GetFitness()))
            #for i in range(len(self.pheromatrix)):
                #print(self.pheromatrix[i])
    
#AntColonyOptimization itself.    
    def AntColonyOptimization(self,migrationcount,colonycount,pheromone_weight,visibility_weight,Q,evaporation):
        #if the given dimension is 1 or lower,the algorithm will not occur,as it does not make sense
        if(self.dimension < 1):
            print("Invalid dimension");
            exit();
        #initialization
        self.visimatrix = self.GetVisibilityMatrix()
        self.pheromatrix=self.GetPheromoneMatrix()
        
        traces = []
        nodecount = len(self.nodelist)
        initcolony = self.GenerateAntColony(colonycount)
        
        
        i=0
        for ant in initcolony:
            nodeindex=np.random.randint(0,len(self.nodelist))
            ant.trajectory.append(self.nodelist[nodeindex])
            #ant.trajectory.append(self.nodelist[i])
            i=i+1
        
        
        for cycle in range(migrationcount):
            colony = copy.deepcopy(initcolony)
            print("Migrating cycle {0}========================================================".format(cycle))
            for ant in colony:
                
                ant.FindPath(pheromone_weight,visibility_weight,self.nodelist,self.pheromatrix,self.visimatrix)
                #print("Ant {0} Length: {1}".format(ant.id,ant.GetFitness()))
                print(str(ant))
            self.UpdatePheromons(colony,evaporation,Q)
            #print("Pheromatrix ".format(cycle))
            #for i in range(len(self.pheromatrix)):
            #    print(self.pheromatrix[i])
            winner=self.GetBestAnt(colony)
            print("Winner Length {0}".format(str(winner.GetFitness())))
            if(len(traces) == 0):
                toappend = copy.deepcopy(winner)
                print("Toappend {0} Length {1}".format(toappend.id,toappend.GetFitness()))
                traces.append(toappend)
            else:
                totalwinner = self.GetBestAnt((traces[-1],winner))
                toappend=copy.deepcopy(totalwinner)
                print("Toappend {0} Length {1}".format(toappend.id,toappend.GetFitness()))
                traces.append(toappend)
            
            
            print("------------------------------------")
                
        for p in traces:
            print(p.GetFitness())
            #drawing the traces
        if(self.dimension == 2):
            #function which actually changes the path
            def UpdatePath(n,points, traces):
                TSPX=[]
                TSPY=[]
                for node in traces[n].trajectory:
                    TSPX.append(node.coors[0])
                    TSPY.append(node.coors[1])
                TSPX.append(traces[n].trajectory[0].coors[0])
                TSPY.append(traces[n].trajectory[0].coors[1])
                print("Drawing "+str(n)+ ". generation with length "+ str(traces[n].GetFitness()))
                points.set_data(TSPX,TSPY)
                return points
            
            fig=plt.figure()
            TSPX=[]
            TSPY=[]
            for node in traces[0].trajectory:
                TSPX.append(node.coors[0])
                TSPY.append(node.coors[1])
            TSPX.append(traces[0].trajectory[0].coors[0])
            TSPY.append(traces[0].trajectory[0].coors[1])
            
            
            points, = plt.plot(TSPX, TSPY, '-o')  
            ani=animation.FuncAnimation(fig, UpdatePath, len(traces), fargs=( points,traces),interval=250,repeat_delay=10000,blit=False)
            plt.show()
            

class NodeListGen:
    def __init__(self):
        pass
    
    def Generate(self,lowbound,upbound,dimension,nodecount):
        self.cities=[]
        j=0
        while(len(self.cities) < nodecount):
            node=Node(j)
            for i in range(dimension):
                node.coors.append(random.SystemRandom().uniform(lowbound,upbound))
            self.cities.append(node)
            j=j+1
        return self.cities
                
                
                
                
cgen = NodeListGen()
clist=cgen.Generate(-20,20,2,300)
#for item in clist:
#    print(str(item))
#print("________________________________________________________")

sol = Solution(2,clist)
#colonycount=5
#colony=sol.GenerateAntColony(colonycount)
#i=0
#for ant in colony:
#    nodeindex=np.random.randint(0,colonycount)
#    ant.trajectory.append(clist[nodeindex])
#    print("Ant {0} starting in node {1}".format(i,nodeindex))
#    i=i+1
#
#for ant in colony:
#    print(str(ant))
#    print("-----------")
sol.AntColonyOptimization(50,20,4,2,1,0.5)
                