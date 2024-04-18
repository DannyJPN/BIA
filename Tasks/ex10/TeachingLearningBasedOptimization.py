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
class ClassMember:
    def __init__(self,lowbound,upbound,fitnessf,coors=[]):
        self.coors=coors
        self.fitnessf=fitnessf
        self.lowbound=lowbound
        self.upbound=upbound

    def __str__(self):
        return "Coors: {0}\nFitness: {1}\n".format(self.coors,self.GetFitness())
        
    def GetFitness(self):
        return self.fitnessf(self.coors)
    
    def MoveAsTeacher(self,totalmean):
        
        r = np.random.uniform(0,1)
        tf = np.random.randint(1,3)
        difference = self.MultiplyVector(r, self.SubtractVectors(self.coors, self.MultiplyVector(tf,totalmean)))
        newcoors = self.AddVectors(self.coors,difference)
        newteacher = ClassMember(self.lowbound,self.upbound,self.fitnessf,self.FitInRange( newcoors,self.lowbound,self.upbound))
        #print("NEW {0}  FOR {1}".format(newteacher.GetFitness(),self.GetFitness()))
        if(newteacher.GetFitness()<=self.GetFitness()):
            #print("BEF {0}".format(self.coors))
            self.coors = newteacher.coors
            #print("AFT {0}".format(self.coors))
            
    def MoveAsStudent(self,mate):
        newcoors=[]
        r = np.random.uniform(0,1)
        if(self.GetFitness() < mate.GetFitness()):
            newcoors = self.AddVectors(self.coors ,self.MultiplyVector(r,self.SubtractVectors(self.coors,mate.coors) ))
        else:
            newcoors = self.AddVectors(self.coors ,self.MultiplyVector(r,self.SubtractVectors(mate.coors,self.coors) ))
        newlearner = ClassMember(self.lowbound,self.upbound,self.fitnessf,self.FitInRange( newcoors,self.lowbound,self.upbound)   )
        if(newlearner.GetFitness()<=self.GetFitness()):
            self.coors =   newlearner.coors
        
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


                
#method responsible for creating the initial classroom of random vectors within predefined boundaries
    def MakeClassroom(self,classroomsize):
        classroom = []
        for i in range(classroomsize):
            ClassMembercoors=[]
            for j in range(self.dimension):
                ranpart=np.random.uniform(self.lowbound, self.upbound)
                ClassMembercoors.append(ranpart)
            classroom.append(ClassMember(self.lowbound,self.upbound,self.fitnessf,ClassMembercoors))
        return classroom
    

#method responsible for selecting the best member of given classroom,based on its fitness.     
    def GetBestMember(self,classroom):
        top = classroom[0]
        for item in classroom:
            if(item.GetFitness() <= top.GetFitness()):
                top=item
        return top
#method responsible for selecting the index of best member of given classroom,based on its fitness,except the indexes specified in toavoid     
    def GetBestMemberIndex(self,classroom,toavoid=[]):
        top = 0
        for i in range(len(classroom)):
            if(classroom[i].GetFitness() <= classroom[top].GetFitness() and i not in toavoid):
                top=i
        return top
        

    
    #method for generating random number except those in toavoid        
    def RandomDifferent(self,max,toavoid=[]):
        rannum=int(random.SystemRandom().uniform(0,max))
        while(rannum in toavoid):
           rannum=int(random.SystemRandom().uniform(0,max))
        return rannum
    
    def GetMean(self,classroom,subject):
        sum=0
        for classmember in classroom:
            sum+= classmember.coors[subject]
        return sum/len(classroom)
        
    def GetTotalMean(self,classroom):
        totalmean=[]
        for i in range(self.dimension):
            totalmean.append(self.GetMean(classroom,i))
        return totalmean
        
#performs and displays TLBO algorithm 
    def TLBO(self,classroomsize,learningcycles,step):
#if the given dimension is lower than 1,the algorithm will not occur,as it does not make sense
        if(self.dimension < 1):
            print("Invalid dimension");
            exit();
        lessons=[]
        Index=[]
        classroom = self.MakeClassroom(classroomsize)
        lessons.append(copy.deepcopy(classroom))
        teachers=[]
        DEX=[]
        DEY=[]
        DEZ=[]
        totalmean = self.GetTotalMean(classroom)
        teacherindex = self.GetBestMemberIndex(classroom)
        teacher = classroom[teacherindex]
        teachers.append(copy.deepcopy(teacher))
   
        for classmember in classroom: 
            print(classmember)
        print("\nINIT STATE  _________________\nTeacher:{0}\nTotalmean: {1}\n________________________\n".format(teacher.GetFitness(),self.fitnessf(totalmean)))
        for cycle in range(learningcycles):
            
            teacher.MoveAsTeacher(totalmean)
            
            for i in range(classroomsize):
                if(i==teacherindex):
                    continue
                randmate = classroom[self.RandomDifferent(len(classroom),[i])]
                classroom[i].MoveAsStudent(randmate)
                
            totalmean = self.GetTotalMean(classroom)
            teacherindex = self.GetBestMemberIndex(classroom)
            teacher = classroom[teacherindex]
            teachers.append(copy.deepcopy(teacher))
            lessons.append(copy.deepcopy(classroom))
            for mem in classroom: 
                Index.append(cycle)
                DEX.append(mem.coors[0])
                DEY.append(mem.coors[1])
                DEZ.append(mem.GetFitness())              
                print(mem)

            
            print("\nMIGRATION CYCLE {0} _________________\nTeacher:{1}\nTotalmean:{2}\n________________________\n".format(cycle,teacher.GetFitness(),self.fitnessf(totalmean)))

            
        dataf = pd.DataFrame({"classroom": Index ,"x" : DEX, "y" : DEY, "z" : DEZ})
        for i in teachers:
            print(i.GetFitness())



#drawing out the algorithm in 3D.
        
        if(self.dimension == 2):
        #function which actually moves the result markers
        

            def UpdatePoint(n,df, points):
                data=df[df['classroom']==n]
                points.set_data(data.x ,data.y)
                points.set_3d_properties(data.z)
                #print("classroom: "+str(list(data.classroom)) + "\tPoint("+str(list(data.x))+ " , "+str(list(data.y))+ ")\twith fitness: "+str(list(data.z)))
                print("classroom: {0}".format(data.classroom))
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
           
            
          
            
            
            data=dataf[dataf['classroom']==0]
            #defining the initial location of result marker
            points, = ax.plot(data.x, data.y, data.z, marker='o',linestyle="")
            # Plot the surface.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.summer,linewidth=0, antialiased=True,alpha=0.3)
            ani=animation.FuncAnimation(fig, UpdatePoint, len(lessons), fargs=(dataf, points),interval=500,repeat_delay=5000)
            
            
            plt.show()

#main    
bfunc=BiaFunc()
sol=Solution(2,bfunc.Sphere,-100,100)
sol.TLBO(5,1500,0.5)
#classroom = sol.MakeClassroom(6)
##
#for s in range(len(classroom)):
#    print("{0}\n{1}".format(s,str(classroom[s])))
##    
#print("____________")
#teacherin = sol.GetBestMemberIndex(classroom)
#teacher = classroom[teacherin]
#print("Leader:\n"+str(teacher))
#print("____________\nBEFORE JUMP")
#
#print(str(classroom[3]))
#
#classroom[3].Jump(teacher,5,0.4,0.6)
#
#print("______\nAFTER JUMP")
#print(str(classroom[3]))



