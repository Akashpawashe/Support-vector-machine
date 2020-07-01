

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sal_train=pd.read_csv("D:\\excelR\\Data science notes\\SVM,CNN,ANN\\asgmnt\\SalaryData_Train(1).csv")
sal_test=pd.read_csv("D:\\excelR\\Data science notes\\SVM,CNN,ANN\\asgmnt\\SalaryData_Test(1).csv")
sal_train.head()
sal_train.describe()
colnames=sal_train.columns


sns.boxplot(x="Salary", y="age", data=sal_train,palette="hls")
sns.boxplot(x="Salary", y="age", data=sal_test,palette="hls")
sns.boxplot(x="Salary", y="workclass", data=sal_train,palette="hls")
sns.boxplot(x="education", y="salary", data=sal_test,palette="hls")

sns.pairplot(data=sal_train)
sns.pairplot(data=sal_test)

string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

from sklearn import preprocessing
LE=preprocessing.LabelEncoder()
for i in string_columns:
    sal_train[i]=LE.fit_transform(sal_train[i])
    sal_test[i]=LE.fit_transform(sal_test[i])
    
    
sal_train["Salary"]=LE.fit_transform(sal_train["Salary"])
sal_test["Salary"]=LE.fit_transform(sal_test["Salary"])	
colnames = sal_train.columns

from sklearn.svm import SVC

trainX= sal_train.iloc[:,0:13]
trainY=sal_train.iloc[:,13]
testX=sal_test.iloc[:,0:13]
testY=sal_test.iloc[:,13]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel= LINEAR
model_linear=SVC(kernel="linear")
model_linear.fit(trainX,trainY)
pred1=model_linear.predict(testX)
np.mean(pred1=testY)

# POLY
model_poly=SVC(kernel="poly")
model_poly.fit(trainX,trainY)
pred2=model_poly.predict(testX)
np.mean(pred2=testY)

# rbf
model_rbf=SVC(kernel="rbf")
model_rbf.fit(trainX,trainY)
pred3=model_rbf.predict(testX)
np.mean(pred3=testY)

# sigmoid
model_sig=SVC(kernel="sigmoid")
model_sig.fit(trainX,trainY)
pred4=model_sig.predict(testX)
np.mean(pred4=testY)

# precomputed
model_pre=SVC(kernel="pecomputed")
model_pre.fit(trainX,trainY)
pred5=model_pre.predict(testX)
np.mean(pred5=testY)
