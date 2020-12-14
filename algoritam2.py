# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:26:54 2020

@author: nikol
"""
import main
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools

X = []
y = []

def getFeatureSet():
    tempList = main.makeFeatureSet()   
    with open('featuresDirty.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|')
        for i in tempList:
            writer.writerow(i)           
        
def readFeatureSet():
    tempList = []
    returnList = []
    with open('features.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            tempList.append(row)
            
    #returnList = [item for sublist in tempList for item in sublist]
    
    return tempList

def getVdata():
    tempList = []
    with open('MovieLensDirty/small/labels-0.1.txt', 'r') as fp:
        reader = csv.reader(fp, delimiter=' ')
        for row in reader:
            tempList.append(row)
            
    return tempList

def readVdata():
    tempList = []
    with open('featuresDirty.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            tempList.append(row)
            
    #returnList = [item for sublist in tempList for item in sublist]
    
    return tempList
    

def bClassifierAccuracy(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print('conf matrix: ', confusion_matrix(y_test, y_pred))
    print('report: ', classification_report(y_test, y_pred))
    return accuracy

def makeX():
    Xtemp = readFeatureSet()
    XtempDirty = readVdata()
    for x1 in Xtemp:
        X.append(x1)
        
    for x2 in XtempDirty:
        X.append(x2)
        

def makeY():
    for i in range(20):
        y.append('0')
        
    for i in range(20):
        y.append('1')

makeX()
makeY()

bClassifierAccuracy(X, y)

def optimalFeatureSetPartition(): #k number of views
    n = len(X[0]) #feature set len from one user
    Xtemp = [[] for i in range(len(X))]
    FA1 = 0
    for i in range(n):
        m = 0
        for j in Xtemp:
            j.append(X[m][i])
            m += 1
        #print(Xtemp)
        print(Xtemp)
        print('##################')
        FA2 = bClassifierAccuracy(Xtemp, y)
        print('acc: ',FA2)
        if FA2 >= FA1:
            FA1 = FA2
        else:
            FA1 = FA2
            for k in Xtemp:
                del k[-1]
    return Xtemp

optimalFeatureSetPartition()
            
        
            
    
            
    
    
    
    
