import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data_svm_org_new_v2.csv')
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values


column=[]
column.append('sample')

for i in range(1792):
    a=list(X[i].split("_"))
    if len(a)==4:
        lhs,rhs=a[3].split(".")
        col=lhs
    elif len(a)==5:
        lhs,rhs=a[4].split(".")
        col=a[3]+"_"+lhs
    elif len(a)==6:
        lhs,rhs=a[5].split(".")
        col=a[3]+"_"+a[4]+"_"+lhs
    if col not in column:
        column.append(col)
    
dataset1=pd.DataFrame(columns=column)
dataset1.fillna(0)


for i in range(1792):
    a=list(X[i].split("_"))
    classe=a[0]+"_"+a[1]+"_"+a[2]+".wav"
            
    dataset1.set_value(i+1,'sample',classe)
    
dataset1=dataset1.drop_duplicates(keep='first')
    

for i in range(1792):
    a=list(X[i].split("_"))
    classe=a[0]+"_"+a[1]+"_"+a[2]+".wav"
    if len(a)==4:
        lhs,rhs=a[3].split(".")
        col=lhs
    elif len(a)==5:
        lhs,rhs=a[4].split(".")
        col=a[3]+"_"+lhs
    elif len(a)==6:
        lhs,rhs=a[5].split(".")
        col=a[3]+"_"+a[4]+"_"+lhs
        
    ind=dataset1.index[dataset1['sample'] == classe]
    dataset1.set_value(ind,col,y[i])

dataset1=dataset1.reset_index()
dataset1=dataset1.drop('index',axis=1)
dataset1.to_csv("dataset.csv",index=False)
