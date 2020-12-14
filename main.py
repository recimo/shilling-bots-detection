import csv
from collections import defaultdict
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import pywt
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import pandas as pd
import math

counter = 0
#incijalizovati izlaz M

# matrica UR
#################################################
def UR_function():
    list1 = []
    with open('MovieLensDirty/small/profiles-0.1.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            if int(row[0]) > 671:
                row.append('September 27, 2020')
                list1.append(row)

    return list1
    #UR = np.asmatrix(list1)


# matrica US
################################################
def US_function():
    list2 = []
    with open('MovieLens/small/ratings.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)
        for row in reader:
            list2.append(row)

    US = np.asmatrix(list2)
    return list2

def sortedListMovieID():
    US = UR_function()
    tempList = []
    tempList2 = []
    for x in US:
        tempList.append(x[1])
    c = Counter(tempList)
    US_dict = dict(c)
    PIS_dict = {k: v for k, v in sorted(US_dict.items(), key=lambda item: item[1], reverse=True)}
    for x in PIS_dict.keys():
        tempList2.append(x)

    return tempList2

def getPIS():
    tempList = sortedListMovieID()
    top20percent = round(len(tempList) / 5)
    PIS = tempList[:top20percent]
    return PIS

def getNIS():
    tempList = sortedListMovieID()
    top20percent = round(len(tempList) / 5)
    NIS = tempList[top20percent:]
    return NIS

def convertTimestampList():
    list = US_function()
    new_list = []
    for x in list:
        time = datetime.fromtimestamp(int(x[3])).strftime("%B %d, %Y")
        templist = x[:3]
        templist.append(time)
        new_list.append(templist)
    return new_list

def TPI(movieID):
    listT = UR_function()
    listTemp = []
    #listMovieIDS = US_function()
    itemList = []
    for x in listT:
        listTemp.append(x[1:2] + x[3:4])
    listTPI = map(tuple, listTemp)
    TPIcount = Counter(listTPI)
    for key in TPIcount.keys():
        if str(movieID) == key[0]:
            tpi_period = []
            tpi_period.append(TPIcount.get(key))
            tpi_period.append(key[1])
            itemList.append(tpi_period)

    return itemList

def DWT(movieID):
    movieList = TPI(movieID)
    (cA, cD) = pywt.dwt(movieList, 'db2')
    y = pywt.idwt(cA, cD, 'db2')
    plt.plot(y)
    plt.ylabel('Test')
    plt.show()
    return y

## treba napisati UT funkciju
##def UT():

def URTPV(userID):
    list = UR_function()
    tempList = []
    tpiList = []
    URTPVector = []
    list_with_dates = UR_function()
    for u in list:
        if str(userID) == u[0]:
            tempList.append(u[1])
    for mid in tempList:
        tpiList = TPI(mid)
    
    i = 0
    for lwd in list_with_dates:
        if lwd[0] == str(userID):
            for tpi in tpiList:
                if lwd[3] in tpi:
                    URTPVector.append(tpi[0])
                    i += 1
                else:
                    URTPVector.append(0)
                    i += 1
        else:
            URTPVector.append(0)
            i += 1
    
    return URTPVector

def IE_URTPV(userID, flag): ### flag ako je 1 uzima sve iz seta, ako ne ne uzima poslednji element
    URTPVect = URTPV(userID)
    binNumber = len(URTPVect)
    tempEntropy = 0
    tempSet = set(URTPVect)
    tempList = list(tempSet)
    if flag == 1:
        for i in tempList:
            currentNumber = URTPVect.count(i)
            tempEntropy = currentNumber / binNumber
            tempEntropy += tempEntropy
    else:
        for i in tempList[:-1]:
            currentNumber = URTPVect.count(i)
            tempEntropy = currentNumber / binNumber
            tempEntropy += tempEntropy
    
    tempEntropy = tempEntropy * -1
    
    return tempEntropy
    
def CALCULATE_CE(userID):
    CE = 0
    E1 = IE_URTPV(userID, 1)
    E2 = IE_URTPV(userID, 0)
    CE = E1 - E2
    
    return CE

### OSTAJE CCE da se izracuna


def R_URTPV(userID):
    list = URTPV(userID)
    result = max(list) - min(list)
    
    return result

def M_URTPV(userID):
    tempSet = URTPV(userID)
    itemSet = set(tempSet)
    suma = 0
    for i in itemSet:
        suma += i
        
    res = suma / len(itemSet)
    
    return res

def V_URTPV(userID):
    urtpv = set(URTPV(userID))
    tempSum = 0
    Murtpv = M_URTPV(userID)
    for i in urtpv:
        tempSum += math.pow((i -  Murtpv),2)
        
    res = 1/len(urtpv)*tempSum
    
    return res

def RDDMA(userID):
    tempSet = URTPV(userID)
    itemSet = set(tempSet)
    dataset = UR_function()
    U = []
    ruit = []
    ti = []
    riti = []
    for u in dataset:
        U.append(u[0])
        
    setU = set(U)
    lenU = len(setU)
    for i in itemSet:
        for ds in dataset:
            if userID == ds[0] and i == ds[1]:
                ruit = ds[2]
                ti.append(ds[3])
                

    tempCnt = 0
    mean_riti = 0
    for t in ti:
        for ds in dataset:
            if t in ds:
                mean_riti += ds[2]
                tempCnt += 1
        mean_riti = mean_riti / tempCnt
        riti.append(mean_riti)
        mean_riti = 0
        tempCnt = 0
        
    cnt = 0
    tempRes = 0
    for m in ruit:
        tempRes += abs(m - riti[cnt])
        cnt += 1
    
    res = 1/len(itemSet)*(tempRes/lenU)    
    
    return res


def WDDMA(userID):
    tempSet = URTPV(userID)
    itemSet = set(tempSet)
    dataset = UR_function()
    U = []
    ruit = []
    ti = []
    riti = []
    for u in dataset:
        U.append(u[0])
        
    setU = set(U)
    lenU = len(setU)
    for i in itemSet:
        for ds in dataset:
            if userID == ds[0] and i == ds[1]:
                ruit = ds[2]
                ti.append(ds[3])
                

    tempCnt = 0
    mean_riti = 0
    for t in ti:
        for ds in dataset:
            if t in ds:
                mean_riti += ds[2]
                tempCnt += 1
        mean_riti = mean_riti / tempCnt
        riti.append(mean_riti)
        mean_riti = 0
        tempCnt = 0
        
    cnt = 0
    tempRes = 0
    for m in ruit:
        tempRes += abs(m - riti[cnt])
        cnt += 1
    
    res = 1/len(itemSet)*(tempRes/pow(lenU,2))    
    
    return res

def IE_URTPV_PIS():
    pis = getPIS()
    binNumber = len(pis)
    tempEntropy = 0
    tempSet = set(pis)
    tempList = list(tempSet)
    for i in tempList:
        currentNumber = pis.count(i)
        tempEntropy = currentNumber / binNumber
        tempEntropy += tempEntropy

    
    tempEntropy = tempEntropy * -1
    
    return tempEntropy


def M_URTPV_PIS():
    tempSet = getPIS()
    itemSet = set(tempSet)
    suma = 0
    for i in itemSet:
        suma += int(i)
        
    res = suma / len(itemSet)
    
    return res
    

def V_URTPV_PIS():
    urtpv = set(getPIS())
    tempSum = 0
    Murtpv = M_URTPV_PIS()
    for i in urtpv:
        tempSum += pow((float(i) -  Murtpv),2)
        
    res = 1/len(urtpv)*tempSum
    
    return res

def M_URP(userID):
    pis = getPIS()
    tempList = UR_function()
    tempSum = 0
    tempI = []
    for u in tempList:
        if str(userID) == u[0]:
            tempI.append(u[1:3])
            
    cntI = 0
    for i in tempI:
        if i[0] in pis:
            tempSum += float(i[1])
            cntI += 1
            
    if cntI == 0:
        cntI += 1
            
    res = tempSum/cntI

    return res

def V_URP(userID):
    pis = getPIS()
    tempList = UR_function()
    tempSum = 0
    tempI = []
    murp = M_URP(userID)
    for u in tempList:
        if str(userID) == u[0]:
            tempI.append(u[1:3])
            
    cntI = 0
    for i in tempI:
        if i[0] in pis:
            tempSum += pow((float(i[1]) - murp), 2)
            cntI += 1
            
    if cntI == 0:
        cntI += 1
            
    res = 1/cntI*tempSum
    
    return res
        

def IE_URTPVN():
    nis = getNIS()
    binNumber = len(nis)
    tempEntropy = 0
    tempSet = set(nis)
    tempList = list(tempSet)
    for i in tempList:
        currentNumber = nis.count(i)
        tempEntropy = currentNumber / binNumber
        tempEntropy += tempEntropy

    
    tempEntropy = tempEntropy * -1
    
    return tempEntropy
    
    
def M_URTPV_NIS():
    tempSet = getNIS()
    itemSet = set(tempSet)
    suma = 0
    for i in itemSet:
        suma += int(i)
        
    res = suma / len(itemSet)
    
    return res

def V_URTPV_NIS():
    urtpv = set(getNIS())
    tempSum = 0
    Murtpv = M_URTPV_NIS()
    for i in urtpv:
        tempSum += pow((float(i) -  Murtpv),2)
        
    res = 1/len(urtpv)*tempSum
    
    return res
    
def M_URN(userID):
    nis = getNIS()
    tempList = UR_function()
    tempSum = 0
    tempI = []
    for u in tempList:
        if str(userID) == u[0]:
            tempI.append(u[1:3])
            
    cntI = 0
    for i in tempI:
        if i[0] in nis:
            tempSum += float(i[1])
            cntI += 1
    if cntI == 0:
        cntI += 1
            
    res = tempSum/cntI

    return res       

def V_URN(userID):
    nis = getNIS()
    tempList = UR_function()
    tempSum = 0
    tempI = []
    murp = M_URN(userID)
    for u in tempList:
        if str(userID) == u[0]:
            tempI.append(u[1:3])
            
    cntI = 0
    for i in tempI:
        if i[0] in nis:
            tempSum += pow((float(i[1]) - murp), 2)
            cntI += 1
            
    if cntI == 0:
        cntI += 1
            
    res = 1/cntI*tempSum
    
    return res

#### TODO: TREBA NAPISATI FUNKCIJU ZA FEATURE SET #####
def makeFeatureSet():
    dataset = UR_function()
    userList = []
    resultList = []
    for u in dataset:
        userList.append(u[0])
        
    userSet = set(userList)
    userSet = sorted(userSet)
    setToList = list(userSet)
    for user in setToList[:20]:
        tempList = []
        tempList.clear()
        tempList.append(IE_URTPV(user, 1))
        tempList.append(R_URTPV(user))
        tempList.append(M_URTPV(user))
        tempList.append(V_URTPV(user))
        tempList.append(RDDMA(user))
        tempList.append(WDDMA(user))
        tempList.append(IE_URTPV_PIS())
        tempList.append(M_URTPV_PIS())
        tempList.append(V_URTPV_PIS())
        tempList.append(M_URP(user))
        tempList.append(V_URP(user))
        tempList.append(IE_URTPVN())
        tempList.append(M_URTPV_NIS())
        tempList.append(V_URTPV_NIS())
        tempList.append(M_URN(user))
        tempList.append(V_URN(user))
        
        resultList.append(tempList)
    
    return resultList       

#****** OVDE POCINJE ALGORITAM 2



