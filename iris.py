# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:12:51 2017

@author: ashirley
"""
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np  
  
#計算距離
def Distance(v1, v2):  
    return np.sqrt(np.sum(np.power(v2 - v1, 2)))    

#初始化中心
def initCentroids(dataSet, k):  
    numSamples, dim = dataSet.shape  
    centroids = np.zeros((k, dim))  
    for i in range(k):  
        # np.random.uniform(low,high,size) [low,high)
        index = int(np.random.uniform(0, numSamples))  
        centroids[i, :] = dataSet[index, :]  
    return centroids  

# k-means 
def kmeans(dataSet, k):  
    numSamples = dataSet.shape[0]  
    # np.zeros 為 150*2 [[0. 0.]...[0. 0.]] 默認為float
    # np.mat 轉為矩陣
    # cluster = [屬於哪個中心, 與中心的距離]
    cluster = np.mat(np.zeros((numSamples, 2)))
    clusterChanged = True  
    
    centroids = initCentroids(dataSet, k)  
  
    while clusterChanged:  
        clusterChanged = False  
          
        for i in range(numSamples):  
            minDist  = 1000.0  
            minIndex = 0               
            for j in range(k):  
                distance = Distance(centroids[j, :], dataSet[i, :]) 
                if distance < minDist:  
                    minDist  = distance  
                    minIndex = j                      
            if cluster[i, 0] != minIndex:  
                clusterChanged = True  
                cluster[i, :] = minIndex, minDist**2    
        # 新中心
        for j in range(k):  
            pointsInCluster = dataSet[np.nonzero(cluster[:, 0].A == j)[0]] 
            centroids[j, :] = np.mean(pointsInCluster, axis = 0)  
            
    return centroids, cluster  

def show(dataSet, k, centroids, cluster):  
    numSamples, dim = dataSet.shape  
    mark = ['or', 'ob', 'og', 'ok', '^r']   
    for i in range(numSamples):  
        markIndex = int(cluster[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])      
    plt.show()  
  
# 讀入鳶尾花資料   
iris = datasets.load_iris()
iris_X = iris.data[:, 2:4]
dataSet = np.mat(iris_X)  
k = int(input("請輸入k值(2~5):"))  
centroids, cluster = kmeans(dataSet, k)  
show(dataSet, k, centroids, cluster)  