from numpy import *                           #Setting the training and testing sets
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
iris=load_iris()
iris.data=iris.data[:100,:2]
iris.data=insert(iris.data,0,1,axis=1)
iris.target=iris.target[:100]
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,train_size=70,random_state=1000)


theta=random.random((3,1))  #initializing parameters,rate and defining the hypothesis function
#print(theta)
rate=0.03
def model(array):
    value=0
    for j in range(3):
        value=value+(array[j]*theta[j][0])
    rvalue=1/(1+exp(-value))
    return rvalue
def output(array):
    value=0
    for j in range(3):
        value=value+(array[j]*theta[j][0])
    if(value>=0):
        return(1)
    else:
        return(0)
def cost(theta):
    sum=0
    for i in range(70):
        sum=sum+(y_train[i]*log(model(x_train[i])))+((1-y_train[i])*log(1-model(x_train[i])))
    return(sum)    


for k in range(10):
    for i in range(70):
        for j in range(3):
            theta[j][0]=theta[j][0]+(rate*(y_train[i]-model(x_train[i]))*x_train[i][j])
        #print("training end",i)
        #print(cost(theta))
    #print("------------------------------------------")    

prediction=zeros(len(y_test))
for i in range(len(y_test)):
    prediction[i]=output(x_test[i])
print("ACCURACY=",mean(y_test==prediction))    
    
