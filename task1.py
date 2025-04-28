import pandas as pd
import numpy as np
import string

def readData(path):
    try:
        data=pd.read_csv(path)
        dataArray=np.array(data)
    except Exception as e:
        print(e)
        data=None
    return dataArray


def calculateDistance(instance1,instance2):
    for k in range(4):
        distance+=instance2[k]-instance1[k]
    
    return np.sqrt(distance)


def trainTestDistance(trainData,testData):
    labels=[]
    l=4
    euclidean=[]
    feature=[]
    for i in range(testData):
        for j in range(trainData):
            distance=calculateDistance(testData[i][1:l],trainData[j][1:l])
            euclidean.append(distance)
            labels.append(trainData[5])
        feature.append(euclidean)
    return feature,labels

def kkn(k,feature,labels):
    samplesLength=len(feature)
    distancesLength=len(feature[0])
    knnFeatures=[]
    minDistance=[]
    knnLabels=[]
    minLabels=[]
    for i in range(samplesLength):
        smallest=0
        count=0
        for j in range(distancesLength):
            if feature[i][j]<=smallest:
                smallest=feature[i][j]
                minDistance.append(smallest)
                minLabels.append(labels)
                count+=1
            if count>=k:
                break
        knnFeatures.append(minDistance)
        knnLabels.append(minLabels)
    return knnFeatures,knnLabels

def getResponse(knnFeatures,knnLabels):
    trainDistances=len(knnFeatures)
    neighbors=len(knnFeatures[0])
    mostOccuring=[]
    CounterOccuring=[]
    for i in range(trainDistances):
        for j in range(neighbors):
            count=0
            clabel=knnLabels[i][j]
            for k in range(neighbors-1):
                if clabel in mostOccuring:
                    
                if knnLabels[i][(j+k)%neighbors]==clabel:
                    count+=1
                if clabel not in mostOccuring:
                    mostOccuring.append(clabel)
            
            CounterOccuring.append(count)
            
                       
            

    return


if __name__=='__main__':
    data=readData('Iris.csv')
    print(data)
    length=len(data)
    index1=int(length*0.8)
    index2=index1+20
    trainData=data[0:index1]
    testData=data[index2:length]
    print('here',trainData[0])
    print(len(trainData))
    print(len(testData))

