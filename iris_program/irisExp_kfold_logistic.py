# !/usr/bin/python

# -*- coding:UTF-8 -*-

'''
@Biao Zhang ||Tiger Zhang
An experiment on data set coiffecient
use 
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression   
from sklearn.ensemble import RandomForestClassifier    
from sklearn import tree    
from sklearn.ensemble import GradientBoostingClassifier    


'''
def multiPercetion(train_data, train_label, test_data, test_label, coiff):

    
    return accuracy
'''

def knnModel(train_data, train_label, test_data, test_label):
    knn = KNeighborsClassifier()  
    knn.fit(train_data, train_label)  
    result = knn.predict(test_data)
    accuracy = 0
    for i in range(0, len(test_label)):
        if test_label[i] == result[i]:
            accuracy = accuracy + 1.0
    accuracy = accuracy / len(test_label)
    return accuracy

def knnModel_withC(train_data, train_label, test_data, test_label, coiff):
    knn = KNeighborsClassifier()  
    knn.fit(np.expm1(coiff * train_data), train_label)  
    result = knn.predict(test_data)  
    accuracy = 0
    for i in range(0, len(test_label)):
        if test_label[i] == result[i]:
            accuracy = accuracy + 1.0
    accuracy = accuracy / len(test_label)
    return accuracy

def svmModel_withC(train_data, train_label, test_data, test_label, coiff):
    clf = svm.SVC()
    clf.fit(np.expm1(coiff * train_data), train_label)
    result = clf.predict(test_data)
    accuracy = 0.0
    for i in range(0, len(test_label)):
        if test_label[i] == result[i]:
            accuracy = accuracy + 1.0
    accuracy = accuracy / len(test_label)
    return accuracy

def svmModel(train_data, train_label, test_data, test_label):
    clf = svm.SVC()
    clf.fit(train_data, train_label)
    result = clf.predict(test_data)
    accuracy = 0.0
    for i in range(0, len(test_label)):
        if test_label[i] == result[i]:
            accuracy = accuracy + 1.0
    accuracy = accuracy / len(test_label)
    return accuracy

def logisticModel(train_data, train_label, test_data, test_label):
    clf = LogisticRegression() 
    clf.fit(np.expm1(coiff * train_data), train_label) 
    result = clf.predict(test_data)
    accuracy = 0.0
    for i in range(0, len(test_label)):
        if test_label[i] == result[i]:
            accuracy = accuracy + 1.0
    accuracy = accuracy / len(test_label)
    return accuracy

def logisticModel_withC(train_data, train_label, test_data, test_label, coiff):
    clf = LogisticRegression() 
    clf.fit(train_data, train_label) 
    result = clf.predict(test_data)
    accuracy = 0.0
    for i in range(0, len(test_label)):
        if test_label[i] == result[i]:
            accuracy = accuracy + 1.0
    accuracy = accuracy / len(test_label)
    return accuracy

def rforestModel(train_data, train_label, test_data, test_label):
    model = RandomForestClassifier(n_estimators=8)    
    model.fit(train_data, train_label)
    result = model.predict(test_data)
    accuracy = 0.0
    for i in range(0, len(test_label)):
        if test_label[i] == result[i]:
            accuracy = accuracy + 1.0
    accuracy = accuracy / len(test_label)
    return accuracy

def rforestModel_withC(train_data, train_label, test_data, test_label, coiff):
    model = RandomForestClassifier(n_estimators=8)    
    model.fit(np.expm1(coiff*train_data), train_label)
    result = model.predict(test_data)
    accuracy = 0.0
    for i in range(0, len(test_label)):
        if test_label[i] == result[i]:
            accuracy = accuracy + 1.0
    accuracy = accuracy / len(test_label)
    return accuracy

def decisionTreeModel(train_data, train_label, test_data, test_label):
    model = tree.DecisionTreeClassifier()    
    model.fit(train_data, train_label)
    result = model.predict(test_data)
    return getAccuracy(result, test_label)

def decisionTreeModel_withC(train_data, train_label, test_data, test_label, coiff):
    model = tree.DecisionTreeClassifier()    
    model.fit(np.expm1(coiff*train_data), train_label)
    result = model.predict(test_data)
    return getAccuracy(result, test_label)

def graBoostingModel(train_data, train_label, test_data, test_label):
    model = GradientBoostingClassifier(n_estimators=200)    
    model.fit(train_data, train_label)
    result = model.predict(test_data)
    return getAccuracy(result, test_label)

def graBoostingModel_withC(train_data, train_label, test_data, test_label, coiff):
    model = GradientBoostingClassifier(n_estimators=200)    
    model.fit(np.expm1(coiff*train_data), train_label)
    result = model.predict(test_data)
    return getAccuracy(result, test_label)    


def getAccuracy(result, test_label):
    accuracy = 0.0
    for i in range(0, len(test_label)):
        if test_label[i] == result[i]:
            accuracy = accuracy + 1.0
    accuracy = accuracy / len(test_label)
    return accuracy


iris_first = load_iris()
#print(iris)
n_samples, n_features = iris_first.data.shape
print((n_samples, n_features))
print(iris_first.data[0])
print(iris_first.target.shape)
print(iris_first.target)
print(iris_first.target_names)
print("feature_names:",iris_first.feature_names)
numlist_1 = np.arange(150)
np.random.shuffle(numlist_1)
iris_temp = iris_first
iris_temp.data = iris_first.data[numlist_1]
iris_temp.target = iris_first.target[numlist_1]
train_data_final_length = 100
train_data_final = iris_temp.data[0:train_data_final_length]
train_label_final = iris_temp.target[0:train_data_final_length]
test_data =  iris_temp.data[train_data_final_length:150]
test_label = iris_temp.target[train_data_final_length:150]

iris_first.data = iris_temp.data[0:train_data_final_length]
iris_first.target = iris_temp.target[0:train_data_final_length]
iris = iris_temp
max_coiff_list = np.array([])
accuracy_final_list = np.array([])
fold_k = range(0, 15)
train_data_len = 80
for k in fold_k:
    numlist = np.arange(train_data_final_length)
    np.random.shuffle(numlist)
    iris.data = iris_first.data[numlist]
    iris.target = iris_first.target[numlist]
    #print train_data
    train_data = iris.data[0:train_data_len]
    #print train_data.shape
    train_label =  iris.target[0:train_data_len]
    #print train_label.shape
    validation_data =  iris.data[train_data_len:train_data_final_length]
    #print validation_data.shape
    validation_label = iris.target[train_data_len:train_data_final_length]
    #print validation_label.shape
   
    coiff_vector = np.arange(0.2, 1.2, 0.001)
    accuracy_vector = np.zeros(np.shape(coiff_vector))
    for i in range(0, len(coiff_vector)):
        coiff = coiff_vector[i]
        accuracy_vector[i] =   logisticModel_withC(train_data, train_label, validation_data, validation_label, coiff)
        #print 'coiffecient = %f, accuracy = %f'%(coiff, accuracy_vector[i])
    print "the %d time..." %(k)  
    posi = np.where(accuracy_vector == np.max(accuracy_vector))
    posi_list = posi[0]
    max_coiff = coiff_vector[posi_list]
    max_coiff_list = np.concatenate((max_coiff_list, max_coiff), axis = 0)
    temp_array = np.max(accuracy_vector) * np.ones(np.shape(max_coiff))
    accuracy_final_list = np.concatenate((accuracy_final_list, temp_array), axis = 0)
    
    '''
    #plot the figure
    plt.plot(coiff_vector, accuracy_vector, '+-')

    plt.xlabel('the coiffecient')
    plt.ylabel('accuracy')

    plt.title("relationship of coifficient and accuracy exp(number*x)-1 ")

    plt.legend()

    plt.show()
    '''
    
#coiff = np.mean(max_coiff_list)
coiff = np.average(max_coiff_list, weights= accuracy_final_list)
print "max accuracy coifficient list"
print max_coiff_list
accuracy_1 = knnModel_withC(train_data_final, train_label_final, test_data, test_label, coiff)
print " knn model accuracy = %f with coiffecient = %f" %(accuracy_1, coiff)
accuracy_2 = knnModel(train_data_final, train_label_final, test_data, test_label)
print " knn model accuracy = %f without coiffecient" %(accuracy_2)

accuracy_1 = svmModel_withC(train_data_final, train_label_final, test_data, test_label, coiff)
print " svm model accuracy = %f with coiffecient = %f" %(accuracy_1, coiff)
accuracy_2 = svmModel(train_data_final, train_label_final, test_data, test_label)
print " svm model accuracy = %f without coiffecient" %(accuracy_2)

accuracy_1 = logisticModel_withC(train_data_final, train_label_final, test_data, test_label, coiff)
print " logistic model accuracy = %f with coiffecient = %f" %(accuracy_1, coiff)
accuracy_2 = logisticModel(train_data_final, train_label_final, test_data, test_label)
print " logistic model accuracy = %f without coiffecient" %(accuracy_2)

accuracy_1 = rforestModel_withC(train_data_final, train_label_final, test_data, test_label, coiff)
print " random forest model accuracy = %f with coiffecient = %f" %(accuracy_1, coiff)
accuracy_2 = rforestModel(train_data_final, train_label_final, test_data, test_label)
print " random forest model accuracy = %f without coiffecient" %(accuracy_2)

accuracy_1 = decisionTreeModel_withC(train_data_final, train_label_final, test_data, test_label, coiff)
print " decision tree model accuracy = %f with coiffecient = %f" %(accuracy_1, coiff)
accuracy_2 = decisionTreeModel(train_data_final, train_label_final, test_data, test_label)
print " decision tree model accuracy = %f without coiffecient" %(accuracy_2)

accuracy_1 = graBoostingModel_withC(train_data_final, train_label_final, test_data, test_label, coiff)
print " gradient boosting model accuracy = %f with coiffecient = %f" %(accuracy_1, coiff)
accuracy_2 = graBoostingModel(train_data_final, train_label_final, test_data, test_label)
print " gradient boosting model accuracy = %f without coiffecient" %(accuracy_2)


'''
1. svm: range:10
knn model accuracy = 0.920000 with coiffecient = 0.366000
 knn model accuracy = 0.940000 without coiffecient
 svm model accuracy = 0.860000 with coiffecient = 0.36600
 svm model accuracy = 0.960000 without coiffecient
 logistic model accuracy = 0.920000 with coiffecient = 0.379453
 logistic model accuracy = 0.680000 without coiffecient
 random forest model accuracy = 0.800000 with coiffecient = 0.379453
 random forest model accuracy = 0.940000 without coiffecient
 decision tree model accuracy = 0.680000 with coiffecient = 0.379453
 decision tree model accuracy = 0.920000 without coiffecient
 gradient boosting model accuracy = 0.560000 with coiffecient = 0.379453
 gradient boosting model accuracy = 0.920000 without coiffecient



2.knn range :15
knn model accuracy = 0.920000 with coiffecient = 0.363039
 knn model accuracy = 0.960000 without coiffecient
 svm model accuracy = 0.960000 with coiffecient = 0.363039
 svm model accuracy = 0.960000 without coiffecient
 logistic model accuracy = 0.940000 with coiffecient = 0.363039
 logistic model accuracy = 0.760000 without coiffecient
 random forest model accuracy = 0.940000 with coiffecient = 0.363039
 random forest model accuracy = 0.940000 without coiffecient
 decision tree model accuracy = 0.960000 with coiffecient = 0.363039
 decision tree model accuracy = 1.000000 without coiffecient
 gradient boosting model accuracy = 0.840000 with coiffecient = 0.363039
 gradient boosting model accuracy = 0.960000 without coiffecient

3. decisonTree: range15
 knn model accuracy = 0.340000 with coiffecient = 0.699500
 knn model accuracy = 0.940000 without coiffecient
 svm model accuracy = 0.300000 with coiffecient = 0.699500
 svm model accuracy = 0.940000 without coiffecient
 logistic model accuracy = 0.960000 with coiffecient = 0.699500
 logistic model accuracy = 0.700000 without coiffecient
 random forest model accuracy = 0.780000 with coiffecient = 0.699500
 random forest model accuracy = 0.940000 without coiffecient
 decision tree model accuracy = 0.340000 with coiffecient = 0.699500
 decision tree model accuracy = 0.920000 without coiffecient
 gradient boosting model accuracy = 0.420000 with coiffecient = 0.699500
 gradient boosting model accuracy = 0.920000 without coiffecient

4. random forest
knn model accuracy = 0.300000 with coiffecient = 0.530840
 knn model accuracy = 0.940000 without coiffecient
 svm model accuracy = 0.380000 with coiffecient = 0.530840
 svm model accuracy = 0.940000 without coiffecient
 logistic model accuracy = 0.920000 with coiffecient = 0.530840
 logistic model accuracy = 0.920000 without coiffecient
 random forest model accuracy = 0.620000 with coiffecient = 0.530840
 random forest model accuracy = 0.920000 without coiffecient
 decision tree model accuracy = 0.880000 with coiffecient = 0.530840
 decision tree model accuracy = 0.920000 without coiffecient
 gradient boosting model accuracy = 0.900000 with coiffecient = 0.530840
 gradient boosting model accuracy = 0.920000 without coiffecient

5. logistic
knn model accuracy = 0.220000 with coiffecient = 0.699500
 knn model accuracy = 0.960000 without coiffecient
 svm model accuracy = 0.400000 with coiffecient = 0.699500
 svm model accuracy = 0.980000 without coiffecient
 logistic model accuracy = 0.940000 with coiffecient = 0.699500
 logistic model accuracy = 0.600000 without coiffecient
 random forest model accuracy = 0.720000 with coiffecient = 0.699500
 random forest model accuracy = 0.960000 without coiffecient
 decision tree model accuracy = 0.380000 with coiffecient = 0.699500
 decision tree model accuracy = 0.940000 without coiffecient
 gradient boosting model accuracy = 0.380000 with coiffecient = 0.699500
 gradient boosting model accuracy = 0.960000 without coiffecient

'''
