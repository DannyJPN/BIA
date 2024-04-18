import numpy as np
from matplotlib import pyplot as plt
import math
import random

bestArray = []

def blindAlg(iterations, fitness, randomFrom, randomTo):

    cycleArray = []
    randomArray = []


    bestNumber = randomTo + 1
    for i in range(iterations):
        cycleArray.append(i)
        #randomNumber = random.randint(randomFrom, randomTo)
        randomNumber = random.uniform(randomFrom, randomTo)
        randomArray.append(randomNumber)

        if(randomNumber < bestNumber):
            bestNumber = randomNumber
            bestArray.append(bestNumber)
        else:
            bestArray.append(bestNumber)


    plt.plot(cycleArray, randomArray, label='Random numbers', linewidth='3')
    plt.plot(cycleArray, bestArray, label='Blind algorithm', linewidth='1')

    plt.title('Blind Algorithm')
    plt.ylabel('Y axis')
    plt.xlabel('X axis')    

    plt.legend()
    plt.show()

blindAlg(20, 0 , 0, 10)
        
        

        
    
