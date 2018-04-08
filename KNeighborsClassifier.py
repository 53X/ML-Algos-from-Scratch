import numpy as np
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,train_size=0.75,random_state=2300)


n_neighbors=3

def knn(n_neighbors,array):
    evaluator=np.zeros(len(x_train))
    collector=np.zeros(n_neighbors)
    k=0
    start=0
    for i in range(len(x_train)):
        evaluator[i]=np.linalg.norm(array-x_train[i])
    for j in range(len(evaluator)):
        if ((evaluator[j]==min(evaluator))&(k<n_neighbors)):
            collector[k]=y_train[j]
            k=k+1
            evaluator[j]=100000000
      
    value=0
    for j in range(n_neighbors):
        counter=0
        for i in range(n_neighbors):
            if (collector[j]==collector[i]):
                counter=counter+1
        if(counter>value):
            value=counter
            classification=collector[j]
    return(classification)


output=np.zeros(len(y_test))
for i in range(len(y_test)):
    output[i]=knn(n_neighbors,x_test[i])

print("ACCURACY=",np.mean(output==y_test))

