### hand written digits classification ######
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC  

plt.close('all')
### lading data
Digits=datasets.load_digits()

images=Digits.images
data=Digits.data
lables=Digits.target
target_names=Digits.target_names

### displaying all 9 class of handwrittings
plt.figure(figsize=(13,5))
for i in range(0,10):
    plt.subplot(4,3,i+1)
    plt.axis('off')
    plt.imshow(images[i,:,:],cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Class: %i' % lables[i],fontsize=10,color='b')
    

### grid search for SVM kernel selection
from sklearn.model_selection import GridSearchCV
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
  {'C': [1, 10, 100, 1000], 'degree': [1,2,3,4,5], 'kernel': ['poly']}
  
 ]
SVM=SVC()
SVM_CV=GridSearchCV(SVM,param_grid,cv=10)
#CV:cross validation

SVM_CV.fit(data,lables)

print(SVM_CV.best_params_)
print(SVM_CV.best_score_)


### spliting data to train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,lables,test_size=0.3,random_state=42,stratify=lables)

### using SVM to classification by rbf kernel
from sklearn.svm import SVC  
#svclassifier = SVC(kernel='poly', degree=8)  
#svclassifier = SVC(kernel='sigmoid')  
#svclassifier = SVC(kernel='linear')  
svclassifier=SVC(gamma=0.001,C=10, kernel='rbf')

svclassifier.fit(x_train, y_train)  
y_pred = svclassifier.predict(x_test) 

### model evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

### showing  4 test images and predicted one
plt.figure()
for i in range(0,4):
    plt.subplot(1,4,i+1)
    plt.axis('off')
    image=np.reshape(x_test[i,:],(8,8))
    plt.imshow(image,cmap=plt.cm.gray_r)
    plt.title('Real and predicted Label: %d %d'%(y_test[i],y_pred[i]))
    
    


    
    




















