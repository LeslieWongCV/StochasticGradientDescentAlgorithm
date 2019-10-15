#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#created at 7.26.18
# debugging and running at 7.31.18
# debugging and running at 8.1.18
"""
Created on Fri Jul 27 13:54:52 2018

@author: leslie
"""

import numpy as np
import matplotlib.pyplot  as plt
import random

#PARAMERTER
#m,n = np.shape(trainData)
maxiteration = 10
lr = 0.01
batch_size = 32
n=100
theta_0 = 10*np.ones(n)

#theta_0 = theta


# DATA GENERRATOR
trainData = []
trainLabel = []
for i in range(1000):
    temp = []
    for j in range(99):
        temp = temp + [random.random()]
    temp = temp  + [1]
    trainData = trainData + [temp]
   # trainLabel = [random.random()]
    trainLabel = trainLabel+ [np.dot(temp,theta_0)]  #trainLable is Related to the trainData


trainData = np.array(trainData)
trainLabel = np.array(trainLabel)
#GENERATED SECOND TIME 
testData = np.array(trainData)
testLabel = np.array(trainLabel)
# ALGORITHM 

def batchGradientDescent (x,y,maxiteration,theta,lr,batch_size):
    losslist = []
    data = []
    for l in range(len(x)):
        data.append(l)
       # data = list(range(len(x))) 
    index = random.sample(data,batch_size) # 获得随机取出部分的index
    # data = list(range(len(x))) !!!!!
    for k in range(0,maxiteration):
        hypothesis = np.dot(x,theta) #预测值是整体求出，反应theta
        loss = hypothesis - y         #loss为全体误差 因为有1000个sample，loss为list类型，即（1000，1）
        lossVal = np.dot(loss,loss)
        losslist.append(lossVal)
        
        #hypothesis and loss  are for the whole trainData
   
        gradient = np.dot(np.take(loss,index),np.take(x,index))/batch_size
        theta = theta - lr*gradient # Gradient Descent Agorilthm 
       # lr = lr   #optimizer HERE 
    return theta, losslist



theta,losslist = batchGradientDescent(trainData,trainLabel,maxiteration,theta_0,lr,batch_size)
plt.plot(losslist)
plt.legend()
plt.show()

for lr in [0.01,0.03]:
    for batch_size in [32,144]:
        theta,losslist = batchGradientDescent(trainData,trainLabel,maxiteration,theta_0,lr,batch_size)
        plt.plot(losslist,label=str(lr)+' batch_size:'+str(batch_size))
#plt.plot(losslist)


#IMPROVE PART 
theta_0 = theta

#PREDICTION 
def predict (x,theta):
    y = np.dot(x,theta)
    return y
###pPRECITTION
y_train = predict(trainData,theta)
y = predict(testData,theta)

plt.plot(y[:30],label='pred')
plt.plot(testLabel[:30],label='test')
plt.legend()
plt.show()


plt.plot(y_train[:30],label='pred')
plt.plot(trainLabel[:30],label='test')
plt.legend()
plt.show()


    