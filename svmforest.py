import pandas as pd 
import numpy as np 
import seaborn as sns

sizeC = pd.read_csv("D:\\excelR\\Data science notes\\SVM,CNN,ANN\\asgmnt\\forestfires.csv")
sizeC.head()
sizeC.describe()
sizeC.columns

x=sizeC.iloc[:,0]
y=sizeC.iloc[:,1]

from sklearn.preprocessing import LabelEncoder
from numpy import array

values = array(x)
print(values)

label_encoder=LabelEncoder()
integerEncoded= label_encoder.fit_transform(values)
print(integerEncoded)

#Converting strings from day to integer
values1 = array(y)
print(values1)

labelEncoder1=LabelEncoder()
integerEncoded1= labelEncoder1.fit_transform(values1)
print(integerEncoded1)

#Dropping columns with strings 
sizeC.drop(["month","day"],axis =1,inplace=True)

#Adding string converted to integer columns to the dataset
df=pd.DataFrame(sizeC)
df['month']=integerEncoded
df['day']=integerEncoded1

sizeC=df
sizeC.columns
sizeC.corr()
sizeC.describe()
sizeC.shape
sizeC.columns

order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,28] # setting column's order
sizeC = df[[df.columns[i] for i in order]]
sizeC.describe()
sizeC.shape


##sns.boxplot(x="lettr",y="x-box",data=sizeC,palette = "hls") 
##sns.boxplot(x="y-box",y="lettr",data=sizeC,palette = "hls") skipped this 2 lines from last pgm just to refer
sns.pairplot(data=sizeC)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(sizeC,test_size = 0.3)
test.head()
train_X = train.iloc[:,:29]
train_y = train.iloc[:,30]
test_X  = test.iloc[:,:29]
test_y  = test.iloc[:,30]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
help(SVC)

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 98.07

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 96.79

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 75.016


