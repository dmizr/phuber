# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:41:13 2020

@author: aiday
"""

import random;
import math;
import numpy as np;
from matplotlib import pyplot as plt

# synthetic data set 2
# suppose 5000 positive inliers, 50000 negative inliers
# 25 positive outliers, 25 negative outliers

inliers = np.random.normal(1, 1, 5000)
inliers = np.append(inliers, np.random.normal(-1, 1, 5000)) 
labels = np.ones([1, 5000]) 
labels = np.append(labels, -1*np.ones([1,5000]))

outliers = np.random.normal(-200, 1, 25)
outliers = np.append(outliers, np.random.normal(200, 1, 25)) 
labels_outliers = np.ones([1, 25]) 
labels_outliers = np.append(labels_outliers, -1*np.ones([1,25]))

# theta goes from -2.0 to 2.0, linear model implies that theta*x is the prediction
def logistic_loss(z):
    return math.log(1+ math.exp(-1*z))

def empirical_risk_logistic_loss_inliers(theta):
    result = 0;
    for i in range(10000) :
        result += logistic_loss(labels[i]*inliers[i]*theta)
    return result/inliers.shape[0]

def empirical_risk_logistic_loss_all(theta):
    result = 0;
    for i in range(10000) :
        result += logistic_loss(labels[i]*inliers[i]*theta)
    for i in range(50) :
        result += logistic_loss(labels_outliers[i]*outliers[i]*theta)
    return result/(inliers.shape[0]+outliers.shape[0])

def huberized_loss(z):
    # technically should be one but we are using 0.5 instead for now
    tau = 0.1
    if (z <= -inverse_sigmoid(tau)):
        return -tau*z - math.log(1-tau)-tau*inverse_sigmoid(tau)
    else :
        return math.log(1+ math.exp(-1*z))
        
def partially_huberized_loss(z):
    # technically should be one but we are using 0.5 instead for now
    tau = 1.1
    if (z <= inverse_sigmoid(1/tau)):
        return -tau/(1+math.exp(-z)) + math.log(tau) + 1
    else :
        return math.log(1+ math.exp(-1*z))
    
def inverse_sigmoid(z):
    return math.log(z/(1-z))
    
def empirical_risk_huberized_loss_inliers(theta):
    result = 0;
    for i in range(10000) :
        result += huberized_loss(labels[i]*inliers[i]*theta)
    return result/inliers.shape[0]

def empirical_risk_huberized_loss_all(theta):
    result = 0;
    for i in range(10000) :
        result += huberized_loss(labels[i]*inliers[i]*theta)
    for i in range(50) :
        result += huberized_loss(labels_outliers[i]*outliers[i]*theta)
    return result/(inliers.shape[0]+outliers.shape[0])

def empirical_risk_partially_huberized_loss_inliers(theta):
    result = 0;
    for i in range(10000) :
        result += partially_huberized_loss(labels[i]*inliers[i]*theta)
    return result/inliers.shape[0]

def empirical_risk_partially_huberized_loss_all(theta):
    result = 0;
    for i in range(10000) :
        result += partially_huberized_loss(labels[i]*inliers[i]*theta)
    for i in range(50) :
        result += partially_huberized_loss(labels_outliers[i]*outliers[i]*theta)
    return result/(inliers.shape[0]+outliers[0])

thetas = np.arange(-2.0, 2.0, 0.05)
logistic_loss_inliers = []
huberized_loss_inliers = []
partially_huberized_loss_inliers = []
logistic_loss_all = []
huberized_loss_all = []
partially_huberized_loss_all = []

for i in thetas:
    logistic_loss_inliers.append(empirical_risk_logistic_loss_inliers(i))
    huberized_loss_inliers.append(empirical_risk_huberized_loss_inliers(i))
    partially_huberized_loss_inliers.append(empirical_risk_partially_huberized_loss_inliers(i))
    logistic_loss_all.append(empirical_risk_logistic_loss_all(i))
    huberized_loss_all.append(empirical_risk_huberized_loss_all(i))
    partially_huberized_loss_all.append(empirical_risk_partially_huberized_loss_all(i))

#plt.figure(figsize=(15,10))
plt.plot(thetas,logistic_loss_inliers, color="green")
plt.plot(thetas,huberized_loss_inliers, color = "red")
plt.plot(thetas,partially_huberized_loss_inliers, color = "blue")
plt.plot(thetas,logistic_loss_all, '--', color="green")
plt.plot(thetas,huberized_loss_all, '--', color="red")
plt.plot(thetas,partially_huberized_loss_all, '--', color="blue")
plt.legend(['Logistic', 'Huber', 'Partial Huber']) 
plt.xlabel(r"$\Theta$")
plt.ylabel(r"R($\Theta$)")
axes = plt.gca()
axes.set_xlim([-2.0,2.0])
axes.set_ylim([0,1.3]) 
#plt.grid(True)
#plt.show()