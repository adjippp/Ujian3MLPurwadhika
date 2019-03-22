import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

df=pd.read_csv('data.csv')
df=df.fillna(np.NaN)
#kita isi df dengan kolom target = 0, target_name = 0 , agar memudahkan untuk training
df['Target']=0
df['Target_name']='Non-Target'
# print(df)
#tandai target dengan angka 1,target_name='Target' pada dataframe usia <= 25, overall >= 80, dan potential >= 80 
df['Target'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]=1
df['Target_name'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]='Target'

x=df.loc[:,['Age','Overall','Potential']]
y=df['Target']
kf=KFold(n_splits = 5)
for train_index,test_index in kf.split(x):
    xtr=x.iloc[train_index]
    ytr=y[train_index]

'''
KNN
nilai k terbaik atau n terbaik dapat dicari dengan cara sqrt(n_data) lalu pilih yg odd/ganjil
cari len dari data (banyak data) lalu kalikan pangkat setengah
'''
k = round(len(x) ** .5)
if((k%2) == 0):
    k=k+1
else:
    k=k
knn=KNeighborsClassifier(n_neighbors=k)

'''
Logistic Regression
'''
logreg=LogisticRegression(multi_class='auto',solver='liblinear')

'''
Random Forest
'''
ranfor=RandomForestClassifier(n_estimators=50)

'''
Decision Tree
'''
dec=DecisionTreeClassifier()
print("Skor KNN: ",round(cross_val_score(knn,xtr,ytr,cv=5).mean()*100),' %')
print("Skor Logistic Regression: ",round(cross_val_score(logreg,xtr,ytr,cv=5).mean()*100),' %')
print("Skor Random Forest: ",round(cross_val_score(ranfor,xtr,ytr,cv=5).mean()*100),' %')
print("Skor Decision Tree: ",round(cross_val_score(dec,xtr,ytr,cv=5).mean()*100),' %')

'''
Skor KNN:  96.0  %
Skor Logistic Regression:  97.0  %
Skor Random Forest:  96.0  %
Skor Decision Tree:  93.0  %
'''