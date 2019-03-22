from sklearn.svm import SVC
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('data.csv')
df=df.fillna(np.NaN)
#kita isi df dengan kolom target = 0, target_name = 0 , agar memudahkan untuk training
df['Target']=0
df['Target_name']='Non-Target'
print(df)
#tandai target dengan angka 1,target_name='Target' pada dataframe usia <= 25, overall >= 80, dan potential >= 80 
df['Target'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]=1
df['Target_name'][(df['Age']<=25)&(df['Overall']>=80)&(df['Potential']>=80)]='Target'

x=df.loc[:,['Age','Overall','Potential']]
y=df['Target_name']

logreg=LogisticRegression(multi_class='auto',solver='liblinear')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=1)
logreg.fit(x_train,y_train)

dfTest=pd.read_csv('datatest.csv')
nilaiX=dfTest.iloc[:,1:]
dfTest['Target']=logreg.predict(nilaiX)
print(dfTest)

'''
                    Name  Age  Overall  Potential      Target
0       Andik Vermansyah   27       87         90  Non-Target
1     Awan Setho Raharjo   22       75         83  Non-Target
2      Bambang Pamungkas   38       85         75  Non-Target
3      Cristian Gonzales   43       90         85  Non-Target
4      Egy Maulana Vikri   18       88         90      Target
5             Evan Dimas   24       85         87      Target
6         Febri Hariyadi   23       77         80  Non-Target
7   Hansamu Yama Pranata   24       82         85      Target
8  Septian David Maulana   22       83         80      Target
9       Stefano Lilipaly   29       88         86  Non-Target
'''