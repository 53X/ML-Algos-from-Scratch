
from numpy import *                                #setting the Training and Testing regions in the dataset (CROSS-VALIDATION)
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=120,random_state=1000)
x_train=insert(x_train,0,1,axis=1)
x_test=insert(x_test,0,1,axis=1)



theta=random.random((3,5))                       #initializing the theta array
theta[2,:]=0

rate=0.03                                        #learning rate


def indicator(element,cls):                      #indicator_function
    if(element==cls):
        return 1
    else:
        return 0

def canonical(cls,train_array):                    #canonical response function
    den=0
    for j in range(3):
        den=den+exp(dot(theta[j,:],train_array))
    num=exp(dot(theta[cls,:],train_array))
    prob=num/den
    return(prob)

def cost(theta_array):                            #log-likelihood function
    training_cost=0
    for i in range(120):
        example_cost=0
        for j in range(3):
            example_cost=example_cost+(indicator(y_train[i],j)*log(canonical(j,x_train[i])))
        training_cost=training_cost+example_cost
    return(training_cost)

def model(test_array):                                    #function returning the obtained class
    output=zeros(3)
    for j in range(3):
        output[j]=canonical(j,test_array)
    classification=max(output)
    for j in range(3):
        if(output[j]==classification):
            classification=j
            break
    return(classification)

        



for k in range(500):                       #training and Stochastic Gradient Ascent
    for i in range(120):
        for j in range(3):
            theta[j,:]=theta[j,:]+(rate*(x_train[i]*(indicator(y_train[i],j)-canonical(j,x_train[i]))))
        #print(cost(theta))
        #print("training end ",i)
    #print("---------------------------------")           
           
prediction=zeros(len(y_test))
for i in range(len(y_test)):
    prediction[i]=model(x_test[i])
    
print("ACCURACY=",mean(y_test==prediction))

