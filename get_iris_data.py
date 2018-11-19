
# Imports
import numpy as np
from numpy import genfromtxt

# Taking iris data from sklearn
from sklearn import datasets
# 
#
from sklearn.model_selection import train_test_split
import sklearn



def buildDataFromIris():
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)
    f=open('cs-training.csv','w')
    for i,j in enumerate(X_train):
        k=np.append(np.array(y_train[i]),j   )
        f.write(",".join([str(s) for s in k]) + '\n')
    f.close()
    f=open('cs-testing.csv','w')
    for i,j in enumerate(X_test):
        k=np.append(np.array(y_test[i]),j   )
        f.write(",".join([str(s) for s in k]) + '\n')
    f.close()
    
    
# Convert to one hot
def convertOneHot(data):
    y=np.array([int(i[0]) for i in data])
    y_onehot=[0]*len(y)
    for i,j in enumerate(y):
        y_onehot[i]=[0]*(y.max() + 1)
        y_onehot[i][j]=1
    return (y,y_onehot)

