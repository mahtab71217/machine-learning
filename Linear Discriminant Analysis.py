###  Linear Discriminant Analysis

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

### loading data
#make_blobs is a function for generating data. In defult it has 100 samples and 2 features
plt.close('all')
Data=datasets.make_blobs()
data=Data[0]
labels=Data[1]

### evaluation of LDA classification method for differents feature/samples ratio
#data generation
def DataGeneration(n_samples,n_features):
    X,Y=make_blobs(n_samples,n_features)
    return X,Y

### spliting data to train and test
n_samples=100
Acc_normal=[]
Acc_shrinakge=[]
ratio=[]

for  n_features in range(1, 150, 4):
    X,Y=DataGeneration(n_samples,n_features)

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=30,random_state=42,stratify=labels)

    ### linear discriminant analysis with shirinkage
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    LDA_normal=LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
    LDA_Shirinkage=LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    LDA_normal.fit(x_train,y_train)
    y_predictNormal=LDA_normal.predict(x_test)
    
    
    LDA_Shirinkage.fit(x_train,y_train)
    y_predictShirinakge=LDA_Shirinkage.predict(x_test)
    
    ratio.append(n_features/n_samples)
    ### model evaluation
    from sklearn.metrics import accuracy_score 
    Acc_normal.append(accuracy_score(y_test,y_predictNormal))
    Acc_shrinakge.append(accuracy_score(y_test,y_predictShirinakge))
    
    
    
plt.plot(ratio,Acc_normal,color='r', linewidth=2,
         label="Linear Discriminant Analysis")

plt.plot(ratio,Acc_shrinakge,color='b',linewidth=2,
         label="Linear Discriminant Analysis with shrinkage")
plt.xlabel('n_features / n_samples')
plt.ylabel('Classification accuracy')
plt.legend(loc=1, prop={'size': 12})
plt.suptitle('Linear Discriminant Analysis vs. \shrinkage Linear Discriminant Analysis ')
   
