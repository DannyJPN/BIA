import numpy as np
from matplotlib import pyplot as plt
import math
import random
import copy
import pandas as pd

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
    def __init__(self, dimension,lowbound,upbound,fitnessf):
        self.dimension=dimension
        self.lowbound=lowbound
        self.upbound=upbound
        self.fitnessf=fitnessf
#method for checking if the vector is within borders 
    def WithinRange(self,vector):
        for i in vector:
            if(i < self.lowbound or i> self.upbound):
                return False
        return True
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
        for i in range(len(x)):
            result.append(x[i]-y[i])
        return result
#method for multiplying a vector with a scalar
    def MultiplyVector(self,A,x):
        result = []
        for i in range(len(x)):
            result.append(A*x[i])
        return result

                
#method responsible for creating the initial generation of random vectors within predefined boundaries
    def MakeGeneration(self,generationsize):
        generation = []
        for i in range(generationsize):
            singlegen=[]
            for j in range(self.dimension-1):
                rannum=np.random.uniform(self.lowbound, self.upbound)
                singlegen.append(rannum)
            generation.append(singlegen)
        return generation
    
#method responsible for selecting the best member of given generation,based on its fitness.     
    def GetBestMember(self,generation):
        top = generation[0]
        for item in generation:
            if(self.fitnessf(item) <= self.fitnessf(top)):
                top=item
        return top
#method responsible for selecting the index of best member of given generation,based on its fitness,except the indexes specified in toavoid     
    def GetBestMemberIndex(self,generation,toavoid=[]):
        top = 0
        for i in range(len(generation)):
            if(self.fitnessf(generation[i]) <= self.fitnessf(generation[top]) and i not in toavoid):
                top=i
        return top
            

    
    #method for generating random number except those in toavoid        
    def RandomDifferent(self,max,toavoid=[]):
        rannum=int(random.SystemRandom().uniform(0,max))
        while(rannum in toavoid):
           rannum=int(random.SystemRandom().uniform(0,max))
        return rannum
#produces Differential Evolution Mutation Vector        
    def MutationVector(self,F,i,generation,selectionmode,subtractioncount):
        #array initialization
        V=np.zeros(self.dimension-1)
        subtracts=[]

        
        randvectors = []
        toavoid=[]
        toavoid.append(i)
        print("Using {0} selection mode with {1} subtractions multiplied by {2}".format(selectionmode,subtractioncount,F))

#random selection strategy
        if(selectionmode == "rand"):
            randvectcount = 2*int(subtractioncount) +1
            #if the neccessary number of random indexes is bigger than the generation length
            if(randvectcount >= len(generation)):
                print("invalid mode,the generation does not have so many members")
                exit()
            print("Generating {0} random indexes".format(randvectcount))
            #getting random vectors x_r1,x_r2 ...
            for r in range(randvectcount):
                ranind = self.RandomDifferent(len(generation),toavoid)
                toavoid.append(ranind)
                randvectors.append(generation[ranind])
                print("R{0} : {1}  :  {2}".format(r+1,ranind,generation[ranind],self.fitnessf(generation[ranind])))
            #performing subtractions,one run of this cycle equals to (x_r - x_r+1)
            for r in range(0,randvectcount-1,2):
                subtracts.append(self.SubtractVectors(randvectors[r],randvectors[r+1]))
            #multiplies each vector by scaling constant. One run of cycle equals to F * s,where s = (x_r - x_r+1) from previous cycle
            for subt in subtracts:
                V = self.AddVectors(V,self.MultiplyVector(F,subt))
            V=self.AddVectors(V,randvectors[-1])
                
#best member strategy  
        elif(selectionmode == "best"):
            randvectcount = 2*int(subtractioncount)
            if(randvectcount >= len(generation)-1):
                print("invalid mode,the generation does not have so many members")
                exit()
            
         
            bestin = self.GetBestMemberIndex(generation,toavoid)
            toavoid.append(bestin)
            bestvector=generation[bestin]
            print("Best  : {0}  :  {1}".format(bestin,generation[bestin],self.fitnessf(generation[bestin])))
            for r in range(randvectcount):
                ranind = self.RandomDifferent(len(generation),toavoid)
                toavoid.append(ranind)
                randvectors.append(generation[ranind])               
                print("R{0} : {1}  :  {2}".format(r+1,ranind,generation[ranind],self.fitnessf(generation[ranind])))
            for r in range(0,randvectcount,2):
                subtracts.append(self.SubtractVectors(randvectors[r],randvectors[r+1]))                

            for subt in subtracts:
                V = self.AddVectors(V,self.MultiplyVector(F,subt))
            V=self.AddVectors(V,bestvector)
        


                
#target to best strategy   
        elif(selectionmode == "target-to-best"):
            randvectcount = 2*int(subtractioncount) -2
            if(randvectcount >= len(generation)-2):
                print("invalid mode,the generation does not have so many members")
                exit()
            currentvector = generation[i]
            print("Target  : {0}  :  {1}".format(i,generation[i],self.fitnessf(generation[i])))
            bestin = self.GetBestMemberIndex(generation,toavoid)
            toavoid.append(bestin)
            bestvector=generation[bestin]
            print("Best  : {0}  :  {1}".format(bestin,generation[bestin],self.fitnessf(generation[bestin])))
            for r in range(randvectcount):
                ranind = self.RandomDifferent(len(generation),toavoid)
                toavoid.append(ranind)
                randvectors.append(generation[ranind])
                print("R{0} : {1}  :  {2}".format(r+1,ranind,generation[ranind],self.fitnessf(generation[ranind])))
            
            
            for r in range(0,randvectcount,2):
                subtracts.append(self.SubtractVectors(randvectors[r],randvectors[r+1]))                
            subtracts.append(self.SubtractVectors(bestvector,currentvector))
            
            for subt in subtracts:
                V = self.AddVectors(V,self.MultiplyVector(F,subt))
            V=self.AddVectors(V,currentvector)        
        
        
        
        else:
            print("Mutation mode unknown")
        

    
        return V
    


    def TrialVector(self,CR,mutatevector,currentvector,crossovermode):
        #crossover mode binomial
        U=[]
        print("Using {0} crossover mode".format(crossovermode))
        if(crossovermode == "bin"):    
            #U = np.zeros(self.dimension-1)
            randindex = np.random.randint(0,self.dimension-1)
            
            for i in range(self.dimension-1):
                rndnum = np.random.uniform(0,1)
                #print("Rannum: {0}".format(rndnum))
                if( rndnum<CR or i ==randindex):
                    U.append( mutatevector[i])
                else:
                    U.append( currentvector[i])
            return U
        #crossover mode exponential
        elif(crossovermode == "exp"):
            separindex = int(len(mutatevector)/2)
            for i in range(separindex):
                U.append(mutatevector[i])
        
            i=0
            while(len(U) < len(currentvector)):
                if(currentvector[i] not in U):
                    U.append(currentvector[i])
            i=i+1
            
        
        else:
            print("Crossover mode unknown")
        return U

#performs and displays DE algorithm 
    def DifferentialEvolution(self,generationcount,generationsize,F,CR,step,strategy="rand/1/bin"):
#if the given dimension is 1 or lower,the algorithm will not occur,as it does not make sense
        if(self.dimension < 2):
            print("Invalid dimension");
            exit();
        gens=[]
        
        strategyparts = strategy.split("/")
        selectionmode= strategyparts[0]
        subtractioncount = strategyparts[1]
        crossovermode = strategyparts[2]
        print("Selection Mode: {0}".format(selectionmode))
        print("Subtractions : {0}".format(subtractioncount))
        print("Crossover Mode: {0}".format(crossovermode))
        generation = self.MakeGeneration(generationsize)
        gens.append(generation)
        DEX=[]
        DEY=[]
        DEZ=[]
        Index=[]
        for g in range(generationcount):
            newgen = copy.deepcopy(generation)
            
            print("START__________________________________________")
            for mem in range(len(newgen)): 
                print("__________________________________________")
                targetvector=generation[mem]
                print("Target: {0}  :   {1}".format(targetvector,sol.fitnessf(targetvector)))
                
                mutatevector = self.MutationVector(F,mem,generation,selectionmode,subtractioncount)
                while(self.WithinRange(mutatevector) == False):
                    mutatevector = self.MutationVector(F,mem,generation,selectionmode,subtractioncount) 
                print("Mutate: {0}  :   {1}".format(mutatevector,sol.fitnessf(mutatevector)))
                trialvector = self.TrialVector(CR,mutatevector,targetvector,crossovermode)
                print("Trial: {0}  :   {1}".format(trialvector,sol.fitnessf(trialvector)))
                acceptvector =  self.GetBestMember((targetvector,trialvector)) 
                newgen[mem] =  acceptvector
                
                print("Accepted: {0}  :   {1}".format(acceptvector,sol.fitnessf(acceptvector)))
            gens.append(newgen)
            generation=newgen
            
            for mem in newgen: 
                print("Generation {0}: {1}  fitness {2}".format(g,mem,self.fitnessf(mem)))
                Index.append(g)
                DEX.append(mem[0])
                DEY.append(mem[1])
                DEZ.append(self.fitnessf(mem))
                
            print("END______________________________________________")
        dataf = pd.DataFrame({"generation": Index ,"x" : DEX, "y" : DEY, "z" : DEZ})

    

#drawing out the algorithm in 3D. 
        if(self.dimension == 3):
        #function which actually moves the result markers
        

            def UpdatePoint(n,df, points):
                data=df[df['generation']==n]
                points.set_data(data.x ,data.y)
                points.set_3d_properties(data.z)
                #print("Generation: "+str(list(data.generation)) + "\tPoint("+str(list(data.x))+ " , "+str(list(data.y))+ ")\twith fitness: "+str(list(data.z)))
                print("Generation: {0}".format(data.generation))
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
           
            
          
            
            
            data=dataf[dataf['generation']==0]
            #defining the initial location of result marker
            points, = ax.plot(data.x, data.y, data.z, marker='o',linestyle="")
            # Plot the surface.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=True,alpha=0.5)
            ani=animation.FuncAnimation(fig, UpdatePoint, len(gens), fargs=(dataf, points),interval=500,repeat_delay=5000)
            
            
            plt.show()

#main    
bfunc=BiaFunc()
sol=Solution(3,-10000,10000,bfunc.Ackley)

sol.DifferentialEvolution(2000,70,0.5,0.5,5,"best/15/bin")

#gener=sol.MakeGeneration(10)
#for g in gener:
#    print("{0} : {1}".format(g,sol.fitnessf(g)))
#
#toavoid=[]
#bestin = sol.GetBestMemberIndex(gener,toavoid)
#print("Bestin: {0}\t{1} : {2}".format(bestin,gener[bestin],sol.fitnessf(gener[bestin])))
#toavoid.append(bestin)
#bestin = sol.GetBestMemberIndex(gener,toavoid)
#print("Bestin: {0}\t{1} : {2}".format(bestin,gener[bestin],sol.fitnessf(gener[bestin])))
#
#print("Bestmem: {0}  :   {1}".format(sol.GetBestMember(gener),sol.fitnessf(sol.GetBestMember(gener))))
#mem=3
#F=2
#CR=0.5
#targetvector=gener[mem]
#mutatevector = sol.MutationVector(F,mem,gener,"rand",1)
#print("Mutate: {0}  :   {1}".format(mutatevector,sol.fitnessf(mutatevector)))
#
#print("Target: {0}  :   {1}".format(targetvector,sol.fitnessf(targetvector)))
#
#trialvector = sol.TrialVector(CR,mutatevector,targetvector,"bin")      
#print("Trial: {0}  :   {1}".format(trialvector,sol.fitnessf(trialvector)))   
#
