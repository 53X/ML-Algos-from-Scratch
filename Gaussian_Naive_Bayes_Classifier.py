
# coding: utf-8

from numpy import *                                #importing dataset and performing cross-validation
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,train_size=130,random_state=1000)
setosa=0
versicolour=0
virginica=0
for i in range(130):
    if(y_train[i]==0):
        setosa=setosa+1
    elif(y_train[i]==1):
        versicolour=versicolour+1
    elif(y_train[i]==2):
        virginica=virginica+1
print("setosa=",setosa)
print("versicolour=",versicolour)
print("virginica=",virginica)
prob_setosa=setosa/130
prob_versicolour=versicolour/130
prob_virginica=virginica/130  
from math import pi

def calc_mean(cls,target_training_array,feature):           #function calculating mean of a given feature in a particular class
    if(cls==0):
        num=42
    elif(cls==1):
        num=44
    elif(cls==2):
        num=44
    array=zeros(num)
    n=0
    for i in range(130):
        if(target_training_array[i]==cls):
            array[n]=x_train[i,feature]
            n=n+1
    return(mean(array))


def calc_var(cls,target_training_array,feature):    #function calculating the variance of a given feature of a particular class
    if(cls==0):
        num=42
    elif(cls==1):
        num=44
    elif(cls==2):
        num=44
    array=zeros(num)
    n=0
    for i in range(130):
        if(target_training_array[i]==cls):
            array[n]=x_train[i,feature]
            n=n+1
    return(var(array))

def prob_total(training_array,cls):               #calculating the probability of an example belonging to a given class
    prob=1
    for j in range(4):
        prob=prob*exp(-((training_array[j]-calc_mean(cls,y_train,j))**2)/(2*calc_var(cls,y_train,j)))/sqrt(2*pi*calc_var(cls,y_train,j))
    if(cls==0):
        return(prob*prob_setosa)
    elif(cls==1):
        return(prob*prob_versicolour)
    elif(cls==2):
        return(prob*prob_virginica)

def model(training_array):                     #returning the class with the largest probability
    x=zeros(3)
    cls=0
    for i in range(3):
        x[i]=prob_total(training_array,cls)
        cls=cls+1
    for i in range(3):
        if(x[i]==max(x)):
            return(i)        


prediction=zeros(len(y_test))
for i in range(len(y_test)):
    prediction[i]=model(x_test[i])

print("ACCURACY=",mean(y_test==prediction))

