#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 09:56:37 2018
实验版本代码
@author: leslie
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import time

random.random()
trainData=[]
trainLabel=[]
for i in range(1000):
    tempd=[]
    for j in range(99):
        tempd=tempd+[random.random()]
    tempd=tempd+[1]   
    trainData=trainData+[tempd]
    trainLabel=trainLabel+[random.random()]
    
trainData=np.array(trainData)
trainLabel=np.array(trainLabel)  

trainData = np.array([[1.1,1.5,1],[1.3,1.9,1],[1.5,2.3,1],[1.7,2.7,1],[1.9,3.1,1],[2.1,3.5,1],[2.3,3.9,1],[2.5,4.3,1],[2.7,4.7,1],[2.9,5.1,1]])
trainLabel = np.array([2.5,3.2,3.9,4.6,5.3,6,6.7,7.4,8.1,8.8])
N=len(trainData)
m, n = np.shape(trainData)
theta_0 = 2*np.ones(n)
theta_real=[0.7149,1.3935,-0.3752]
alpha = 0.1 #LR 
m = 2
maxIteration = 25



def batchGradientDescent(data, label, theta, alpha, m, maxIterations,mode=0):
    xTrains = data.transpose()
    losslist=[]
    #data = []

    #for i in range(10):
    #    data.append(i)
    for i in range(0, maxIterations):
        hypothesis = np.dot(data, theta)
        loss = np.dot(hypothesis - label,hypothesis - label)/ N/2
        #print (loss)
        losslist.append(loss)  #用losslisthold住每次的loss值 
        #xTrains = random.sample(data,m)
        gradient = np.dot(xTrains, hypothesis-label) / m
        theta = theta - alpha * gradient
        if mode==1:
            alpha=0.98*alpha
        #print (theta)
    return theta,losslist

def predict(data, theta):  #用theta预测
    m, n = np.shape(data)
    xTest = np.ones((m, n+1))
    xTest[:, :-1] = x
    label_Prediction = np.dot(xTest, theta)
    return label_Prediction
#!!!!!!!!!
for alpha in [0.03,1]:
    for mode in range(2):
        theta,losslist = batchGradientDescent(trainData, trainLabel, theta_0, alpha, m, maxIteration,mode)
        plt.plot(losslist,label=str(alpha)+'#'+str(mode))
        
plt.legend()
plt.show()


alpha=0.01
theta,losslist = batchGradientDescent(trainData, trainLabel, theta_0, alpha, m, maxIteration)
plt.plot(losslist,label=str(alpha))
plt.legend()
plt.show()
#test
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
y = np.array([[9.5],[10.2],[10.9],[11.6],[12.3]])# supposed to be 
#print (predict(x, theta))
print(predict(x,theta))


#-------------------------*--------------------*_-----------------*------------

# SGD 

def StochasticGradientDescent(x_data, y, theta, alpha, m, maxIterations,mode=0):
     data = []
     losslist=[]
     
     for i in range(len(x_data)):
         data.append(i)
   #  xTrains = x.transpose()
     for i  in range(0,maxIterations):
         hypothesis = np.dot(x_data,theta)
         #loss =hypothesis - y
         loss = hypothesis - y
         loss_2 = np.dot(hypothesis - y,hypothesis - y)/2
         losslist.append(loss_2)
                
         #index = random.sample(data,1)#*    list形式
         index = random.sample(data,m)
        # index1 = index[0]          #int形式
         
        # gradient = loss[index1]*x_data[index1] 
         gradient = np.matmul(np.take(x_data,index,axis=0) .transpose(),np.take(loss,index))/m
         #print(gradient)
         theta = theta - alpha * gradient #*
         if mode==1:
             alpha=0.98*alpha
         
         
     return theta,losslist

 


#test
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
y = np.array([[9.5],[10.2],[10.9],[11.6],[12.3]])# supposed to be 

for alpha in  [0.001]:
    for m in [1,5,1000]:
        start=time.time()
        theta,losslist = StochasticGradientDescent(trainData, trainLabel, theta_0, alpha, m, maxIteration)
        end=time.time()
        
        plt.plot(losslist,label=str(alpha)+ '#'+str(m)+'time='+str(1*(end-start)))

plt.legend()
plt.show()

print(predict(x,theta))

'''
trainlabelo = []
TT = []
for h in range(1000):
    tempo = []
   # tempo = tempo + [random.random()]
    for o in range(99):
        tempo = tempo + [random.random()]
    tempo = tempo +[1]
    TT = TT + [tempo]
    
    trainlabelo = trainlabelo + [random.random()]
TT=np.array(TT)##!!!!


trainData = []
trainLabel = []
for i  in range (1000):
    temp = []
    for j  in range (100):
        temp = temp + [random.random()]
    trainData = trainData + [temp]
    
trainData = trainData + [temp]
temp = [1,2,3123,1,2,12,399,312,312,1]
trainData=[]
trainLabel=[]
for i in range(1000):
    tempd=[]
    for j in range(99):
        tempd=tempd+[random.random()]
    tempd=tempd+[1]   
    trainData=trainData+[tempd]
    trainLabel=trainLabel+[random.random()]
    '''

