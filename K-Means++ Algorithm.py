import matplotlib
k=3
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
from numpy import *
from sklearn.datasets import make_blobs
dataset=make_blobs(centers=k,n_samples=200,n_features=2,random_state=1000)
data=dataset[0]
centroid=zeros((k,2))

distance_array=zeros(len(data))


def centroid_calculator(centroid_array):
    for i in range(len(data)):
        distance_current=zeros(len(centroid_array))
        for k in range(len(centroid_array)):
            distance_current[k]=linalg.norm(data[i]-centroid_array[k])
        distance_array[i]= max(distance_current)   
    for i in range(len(data)):
        if(distance_array[i]==max(distance_array)):
            return(data[i])
    

def centroid_initialize():
    centroid_array=[]
    for i in range(k):
        if(i==0):
            first_choice=random.randint(0,200)
            print("FIRST RANDOM CHOICE NO.=",first_choice)
            centroid[i]=array((data[first_choice]))
            centroid_array=insert(centroid_array,i,centroid[i],axis=0)
        else:
            centroid[i]=centroid_calculator(centroid_array)
            centroid_array=insert(centroid_array,i,centroid[i],axis=0)
    return(centroid)        

centroid=centroid_initialize()
print("CENTROID=",centroid)

cluster_evaluator=zeros(k)
cluster=zeros(200)

def cluster_eval():
    for i in range(200):
        for j in range(k):
            cluster_evaluator[j]=linalg.norm(data[i]-centroid[j])
        for j in range(k):
            if(cluster_evaluator[j]==min(cluster_evaluator)):
                cluster[i]=j
    return(cluster)

def centroid_upd():
    for j in range(k):
        count=0
        val=0
        for i in range(200):
            if(cluster[i]==j):
                count=count+1
                val=val+data[i]
        centroid[j]=val/count
        return(centroid)
    
def distortion():
    s=0
    for i in range(200):
        s=s+(linalg.norm(data[i]-centroid[cluster[i]])**2)
    return(s)

sentinel_prev=distortion()
sentinel=0
for g in range(20):
    sentinel_prev=sentinel
    cluster=cluster_eval()
    centroid=centroid_upd()
    sentinel=distortion()
    print(sentinel)
    for i in range(200):
        if(cluster[i]==0):
            plt.scatter(data[i,0],data[i,1],facecolor='cyan',edgecolor='black')
        elif(cluster[i]==1):
            plt.scatter(data[i,0],data[i,1],color='red',edgecolor='black')
        elif(cluster[i]==2):
            plt.scatter(data[i,0],data[i,1],color='green',edgecolor='black')
    plt.show() 


labels=dataset[1]
plt.scatter(data[:,0],data[:,1],c=labels)
plt.show()


