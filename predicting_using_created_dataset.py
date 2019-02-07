from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#importing metric that here is accuracy_score
from sklearn.metrics import accuracy_score
import pandas as pd

dataset=pd.read_csv("dataset.csv")

X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

i=0
for i in range(128):
    a=list(y[i].split("_"))
    dataset=dataset.set_value(i,'sample',a[1]+"_"+a[2])

#converting categorical data to Numerical
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataset.iloc[:,0] = label_encoder.fit_transform(dataset.iloc[:,0].values)
y=dataset.iloc[:,0].values

#splitting data in 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#applying LogisticRegression
lr = LogisticRegression(C=10.0)
lr.fit(X_train, y_train)

#applying Random Forest Classsifier
rf = RandomForestClassifier(max_depth=None)
rf.fit(X_train, y_train)

#Applying SVM
svc = SVC(C=5.0)
svc.fit(X_train, y_train)

column=list(dataset.columns.values)
column.remove('sample')

value=[]
value1=[]
value2=[]

#Predicting values
a=lr.predict(X_test)
b=rf.predict(X_test)
c=svc.predict(X_test)

#Converting numerical to categorical for spacing what class does prediction belong to
for i in a:
    if i==0:
        value.append("babble_sn10")
    elif i==1:
        value.append("babble_sn5")
    elif i==2:
        value.append("car_sn10")
    elif i==3:
        value.append("car_sn5")
    elif i==4:
        value.append("street_sn10")
    elif i==5:
        value.append("street_sn5")
    elif i==6:
        value.append("train_sn10")
    elif i==7:
        value.append("train_sn5")

for i in b:
    if i==0:
        value1.append("babble_sn10")
    elif i==1:
        value1.append("babble_sn5")
    elif i==2:
        value1.append("car_sn10")
    elif i==3:
        value1.append("car_sn5")
    elif i==4:
        value1.append("street_sn10")
    elif i==5:
        value1.append("street_sn5")
    elif i==6:
        value1.append("train_sn10")
    elif i==7:
        value1.append("train_sn5")

for i in c:
    if i==0:
        value2.append("babble_sn10")
    elif i==1:
        value2.append("babble_sn5")
    elif i==2:
        value2.append("car_sn10")
    elif i==3:
        value2.append("car_sn5")
    elif i==4:
        value2.append("street_sn10")
    elif i==5:
        value2.append("street_sn5")
    elif i==6:
        value2.append("train_sn10")
    elif i==7:
        value2.append("train_sn5")

value3=[]
for i in y_test:
    if i==0:
        value3.append("babble_sn10")
    elif i==1:
        value3.append("babble_sn5")
    elif i==2:
        value3.append("car_sn10")
    elif i==3:
        value3.append("car_sn5")
    elif i==4:
        value3.append("street_sn10")
    elif i==5:
        value3.append("street_sn5")
    elif i==6:
        value3.append("train_sn10")
    elif i==7:
        value3.append("train_sn5")


#Comparing predicted vs actual
print()
print("-----------------------------------------")
print("|     Logistic Regression Predictions   |")
print("-----------------------------------------")
for i in  range(26):
    print()
    print("Predicted: ",value[i]," ","Actual: ",value3[i])
    

print()
print("-----------------------------------------")
print("|Random Forest Classification Predictions|")
print("-----------------------------------------")

for i in  range(26):
    print()
    print("Predicted: ",value1[i]," ","Actual: ",value3[i])


print()
print("-----------------------------------------")
print("|             SVM Predictions           |")
print("-----------------------------------------")

for i in  range(26):
    print()
    print("Predicted: ",value2[i]," ","Actual: ",value3[i])


print("-----------------------------------------")
print("|             Final Result              |")
print("-----------------------------------------")
print()
print("Logistic Regression: ", accuracy_score(y_test, a)*100)
print()
print("Random Forest Classifier: ", accuracy_score(y_test,b)*100)
print()
print("Support Vector Machine: ", accuracy_score(y_test,c)*100)
print()