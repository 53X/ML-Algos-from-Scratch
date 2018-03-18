
#specifying the test and training set
from numpy import *
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data[0:100,:]
y=iris.target[0:100]
for i in range(100):
    if (y[i]==0):
        y[i]=-1
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=70,random_state=1000)



alpha=zeros(70)                     #initialization of lagrange multipliers, threshold and regularization parameter
regularization_parameter=10^-5

def kkt_checker(number):            #func showing whether a particular training example follows the KKT conditions
    sum=0
    beta=threshold()
    for j in range(70):
        sum=sum+alpha[j]*y_train[j]*dot(x_train[j],x_train[number])
        sum=sum+beta
        sum=sum*y_train[number]
   
    if(alpha[number]==0):
        if(sum>=1):
            return(1)
        else:
            return(0)
        
    if(alpha[number]==regularization_parameter):
        if(sum<=1):
            return(1)
        else:
            return(0)
                
    if(alpha[number]>0 and alpha[number]<regularization_parameter):
        if(sum==1):
            return(1)
        else:
            return(0)
        
def error(index):
    value=0
    add=0
    intercept=threshold()
    for i in range(70):
        add=add+alpha[i]*y_train[i]*dot(x_train[i],x_train[index])
    add=add+intercept
    value=add-y_train[index]
    return(value)


def selection_updation():                   #selection of alpha_1 and alpha_2 lagrange multipliers
    alpha_1=0
    alpha_2=0
    for i in range(70):
        if (kkt_checker(i)==0):
            alpha_1=alpha[i]
            break
    result_1=y_train[i]
    
    j=random.randint(0,70)
   
    if(j==i):
        j=random.randint(0,70)
    alpha_2=alpha[j]
    result_2=y_train[j]
   
    lower_bound,upper_bound=0,0
    
    if(result_1 != result_2):
        lower_bound=max(0,alpha_2-alpha_1)
        upper_bound=min(regularization_parameter,regularization_parameter+alpha_2-alpha_1)
    else:
        lower_bound=max(0,alpha_1+alpha_2-regularization_parameter)
        upper_bound=min(regularization_parameter,alpha_1+alpha_2)
        
    alpha_2_old,alpha_1_old=alpha_2,alpha_1
    alpha_2=alpha_2-(y_train[j]*(error(i)-error(j))/(2*dot(x_train[i],x_train[j])-dot(x_train[i],x_train[i])-dot(x_train[j],x_train[j])))   

    
    if(alpha_2>upper_bound):
        alpha_2=upper_bound
    elif(alpha_2<lower_bound):
        alpha_2=lower_bound
    elif(alpha_2>=lower_bound and alpha_2<=upper_bound):
        alpha_2=alpha_2
    
    alpha_1=alpha_1+(y_train[i]*y_train[j]*(alpha_2_old-alpha_2))
    alpha[i]=alpha_1
    alpha[j]=alpha_2


def theta():                                                 #the parameter W
    add=0
    for i in range(70):
        add=add+(alpha[i]*y_train[i]*x_train[i])
    return(add)    


def threshold():                                            #the threshold parameter/intercept parameter ,b
    value1=0
    value2=0
    parameter=theta()
    for i in range(70):
        if(y_train[i]==-1):
            max1=dot(parameter,x_train[i])
            if(max1>=value1):
                value1=max1
        if(y_train[i]==1):
            min1=dot(parameter,x_train[i])
            if(min1<=value2):
                value2=min1
    result=(value1+value2)/2
    return(result)

            
def model(array):                                         #he_SVM_model
    w=theta()
    b=threshold()
    value=dot(w,array)+b
    if(value>=0):
        return(-1)
    elif(value<0):
        return(1)
    

for k in range(1400):
    selection_updation()
    #print(alpha)
    #print("end of training",k)
prediction=zeros(len(y_test))
for i in range(len(y_test)):
    prediction[i]=model(x_test[i])
    
print("ACCURACY=",mean(y_test==prediction))

